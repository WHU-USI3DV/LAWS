import time
import os
import sys
import torch
import logging
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.model_factory import model_factory
# from util import parser
from util.scannet_config import Config
from datasets.test_dataset import ScannetPRTestDataset, ScannetTripleCollate
import numpy as np
import pickle
import csv
import math
from models.MinkLoc3dv2.mink_params import make_sparse_tensor_rgb
from sklearn.neighbors import KDTree
# args = parser.parse_arguments()
from eval.eval_config import get_config_eval
import MinkowskiEngine as ME
from models.MinkLoc3dv2.mink_params import CartesianQuantizer

args = get_config_eval()

def get_point_input(pointclouds):
    pointclouds = torch.from_numpy(pointclouds).unsqueeze(0).type(torch.FloatTensor).to("cuda")
    return pointclouds            
    
global quantizer 
quantizer = CartesianQuantizer(quant_step=0.01)

# def get_sparse_input(pointclouds):
#     pointclouds = torch.from_numpy(pointclouds)
#     xyz, rgb = pointclouds[:,:3].float(), pointclouds[:,3:].float()
#     coords, feats = ME.utils.sparse_collate([xyz], [rgb])
#     batch = {'coords': coords, 'features': feats}
#     batch = {e: batch[e].to('cuda') for e in batch}
    
#     return batch

global quantize_step
quantize_step = 0.01

def get_sparse_input(pointclouds):
    pointclouds = torch.from_numpy(pointclouds)
    num_points = pointclouds.shape[0]
    xyz, rgb = pointclouds[:,:3].float(), pointclouds[:,3:].float()
    label = torch.ones((num_points, 1),dtype=torch.int32)
    # Quantize the input
    coords, feats = [], []
    discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
        coordinates=xyz,
        features=rgb,
        labels=label,
        quantization_size=quantize_step,
        ignore_label=-100)
    coords.append(discrete_coords)
    feats.append(unique_feats)
    ##########################################
    coords_batch, feats_batch = [], []
    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    batch = {'coords': coords_batch, 'features': feats_batch} 
    batch = {e: batch[e].to('cuda') for e in batch}
    return batch 
    
def evaluate_checkpoint(model, save_path, cfg):
    # try:
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Use pretrain model')
    return model, checkpoint['epoch']

def evaluate_checkpoint_laws(model, save_path, cfg):
    # try:
    print('Use pretrain model')
    state = torch.load(save_path)  # ,map_location='cuda:0')
    epoch = state['epoch'] + 1
    #### Updating 
    for i in range(epoch):
      if i < cfg.groups_num:
          model.update_aggregators()
    #### Loading
    model.load_state_dict(state['model_state_dict'])
    
    return model, state['epoch']

def get_yaw(pose):
    
    sy = np.sqrt(pose[0,0]**2 + pose[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2( pose[2,1], pose[2,2])
        y = math.atan2(-pose[2,0], sy)
        z = math.atan2( pose[1,0], pose[0,0])
    else:
        x = math.atan2(-pose[1,2], pose[1,1])
        y = math.atan2(-pose[2,0], sy)
        z = 0
    roll = math.degrees(x)
    pitch = math.degrees(y)
    yaw = math.degrees(z)
    return roll, pitch, yaw
    
def evaluate():
    print('\nScannet Indoor Place Recognition...\n')
    ''' === Load Model and Backup Scripts === '''
    #### Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = args.checkpoint_name
    print('Loading checkpoint from: ', save_path)
    model = model_factory(args)  
    if args.eval_mode=='laws':
        model, epoch = evaluate_checkpoint_laws(model, save_path, args)
    elif args.eval_mode=='cp' or args.eval_mode=='ret':
        model, epoch = evaluate_checkpoint(model, save_path, args)
    model = model.to(device)
    model.eval()    
      
    if args.backbone in ['mink', 'mink_laws']:
        get_input = get_sparse_input
    else:
        get_input = get_point_input
    # Initialise and Load the configs
    # dataset config
    dataset_config = Config()
    # # Change parameters for the TESTing here. 
    dataset_config.validation_size = 3700    # decide how many points will be covered in prediction -> how many forward passes
    # config.input_threads = 0
    # dataset_config.print_current()

    print('\nData Preparation')
    print('****************')
    t = time.time()
    # print('Test data:')
    # new dataset for triplet input
    test_dataset = ScannetPRTestDataset(dataset_config, 
                                        args.scannetpr_dir,
                                        set='test')
    
    # Initialize the dataloader
    test_loader = DataLoader(test_dataset, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=0,
                            collate_fn=ScannetTripleCollate, 
                            pin_memory=True)
    dataloader_iterator = iter(test_loader) 
    print('Done in {:.1f}s\n'.format(time.time() - t))

    save_path, checkpoint = os.path.split(args.checkpoint_name)
    
    db_path = os.path.join(save_path, 'database_'+str(epoch))
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    start = time.time()
    dist_thred = 3.0
    # load database point clouds from file
    vlad_file = os.path.join(db_path, 'vlad_KDTree.txt')
    bIdfId_file = os.path.join(db_path, 'file_id.txt')
    bIdbId_file = os.path.join(db_path, 'batch_id.txt')
    if not os.path.exists(vlad_file):
        print('\nCreating database')
        print('*******************',len(test_loader))
        t = time.time()
        # Get database
        break_cnt = 0
        database_vect = []
        batchInd_fileId = []
        batchInd_batchId = []
        database_cntr = {}
        db_count = 0
        for i in range(len(test_loader)):
            # continue if empty input list is given
            # caused by empty positive neighbors
            batch = next(dataloader_iterator)
            # print(batch.points.shape)
            # print(batch.frame_centers, batch.world_crt, batch.frame_inds)
            
            if len(batch.points) == 0:
                break_cnt +=1
                continue
            else:
                break_cnt = 0
            # stop fetch new batch if no more points left
            if break_cnt > 4:
                break

            ## NOTE centroid here is zero meaned. Use un-meaned pts for centroid test
            tmp_cntr = batch.world_crt    # np.array, (3,)
            tmp_fmid = batch.frame_inds       # list, [scene index, frame index]
           
            if tmp_fmid[0] not in database_cntr.keys():
                # print('- ADDING NEW PCD TO DB:', tmp_fmid, db_count)
                # batch.to(device)
                # get the VLAD descriptor
                pointclouds = np.concatenate((batch.points, batch.features), axis=-1)
                # pointclouds = torch.from_numpy(pointclouds).unsqueeze(0).type(torch.FloatTensor).to("cuda")
                
                pointclouds = get_input(pointclouds)
                
                descriptors = model(pointclouds, is_training=False)    
                
                # store vlad vec, frm_cntr, and indices
                database_vect.append(descriptors.cpu().detach().numpy()[0]) # append a (1,256) np.ndarray
                database_cntr[tmp_fmid[0]] = [tmp_cntr]
                batchInd_fileId.append(tmp_fmid)
                batchInd_batchId.append(i)

                db_count += 1

            else:
                # initialise boolean variable
                bAddToDB = True

                ## Only check with distance threshold
                for db_cntr in database_cntr[tmp_fmid[0]]:
                    tmp_dist = np.linalg.norm(db_cntr - tmp_cntr)
                    if tmp_dist < dist_thred:
                        # skip if not enough movement detected
                        bAddToDB = False
                        break

                if bAddToDB:
                    # print('- ADDING NEW PCD TO DB:', tmp_fmid, db_count)
                    pointclouds = np.concatenate((batch.points, batch.features), axis=-1)
                    pointclouds = get_input(pointclouds)
                    descriptors = model(pointclouds, is_training=False)  
                    
                    # store vlad vec, frm_cntr, and indices
                    database_vect.append(descriptors.cpu().detach().numpy()[0]) # append a (1,256) np.ndarray
                    database_cntr[tmp_fmid[0]].append(tmp_cntr)
                    batchInd_fileId.append(tmp_fmid)
                    batchInd_batchId.append(i)

                    db_count += 1
        
            # print('stored center number:', len(database_cntr[tmp_fmid[0]]))
        database_vect = np.array(database_vect)

        print('DB size:', db_count, database_vect.shape)
        search_tree = KDTree(database_vect, leaf_size=4)
        # print(batchInd_fileId)
        # print(database_vect.shape)

        # store the database
        # with open(vlad_file, "wb") as f:
        #     pickle.dump(search_tree, f)
        # with open(bIdfId_file, "wb") as f:
        #     pickle.dump(batchInd_fileId, f)
        # with open(bIdbId_file, "wb") as f:
        #     pickle.dump(batchInd_batchId, f)
        # print('VLAD Databased SAVED to Files:', os.path.join(db_path, 'XXXX.txt'))

    else:
        # load the database
        # store the database
        with open(vlad_file, "rb") as f:
            search_tree = pickle.load(f)
        with open(bIdfId_file, "rb") as f:
            batchInd_fileId = pickle.load(f)
        with open(bIdbId_file, "rb") as f:
            batchInd_batchId = pickle.load(f)
        print('VLAD Databased LOADED from Files:', os.path.join(db_path, 'XXXX.txt'))
        db_vlad_vecs = np.array(search_tree.data, copy=False)
        print('Total stored submaps are:', db_vlad_vecs.shape)

    print('Done in {:.1f}s\n'.format(time.time() - t))


    print('\nStart test')
    print('**********')
    t = time.time()
    # loop again to test with KDTree NN
    break_cnt = 0
    test_pair = []
    eval_results = []
    query_lists = []
    log_strings = ''
    for i, batch in enumerate(test_loader):
        # continue if empty input list is given
        # caused by empty positive neighbors
        if len(batch.points) == 0:
            break_cnt +=1
            continue
        else:
            break_cnt = 0
        # stop fetch new batch if no more points left
        if break_cnt > 4:
            break

        # print('processing pcd no.', i)

        # skip if it's already stored in database
        # if i in batchInd_batchId or i%30 != 0:  # every 450 frame 
        if i in batchInd_batchId:
            continue

        tt = time.time()
        q_fmid = batch.frame_inds     # list, [scene_id, frame_id]
        # Get VLAD descriptor
        pointclouds = np.concatenate((batch.points, batch.features), axis=-1)
        pointclouds = get_input(pointclouds)
        vlad = model(pointclouds, is_training=False)     
        vlad = vlad.cpu().detach().numpy()
        
        dist, ind = search_tree.query(vlad, k=3)
        test_pair.append([q_fmid, 
            [batchInd_fileId[ ind[0][0] ],
                batchInd_fileId[ ind[0][1] ],
                batchInd_fileId[ ind[0][2] ]]
        ])

        q_cent = batch.world_crt    # np.array, (3,)
        # q_pcd_np = batch.points[0].cpu().detach().numpy()         # np.ndarray, (n, 3)
        
        q_pose = test_loader.dataset.poses[q_fmid[0]][q_fmid[1]]
        q_camera_ctr = batch.frame_centers
        q_camera_ctr = q_pose[:3, 3]
        roll, pitch, yaw = get_yaw(q_pose)
        
        one_result = []
        top1_fmid = batchInd_fileId[ind[0][0]]
        for k, id in enumerate(ind[0]):
            r_fmid = batchInd_fileId[id]
            log_strings += ('--' + str(r_fmid[0]) + '_' + str(r_fmid[1]))
            
            # get k-th retrieved point cloud
            retri_file = test_loader.dataset.files[r_fmid[0]][r_fmid[1]]
            retri_file = retri_file.split('input_pcd_0mean')
            
            if r_fmid[0] != q_fmid[0]:
                one_result.append(0)
                log_strings += ': FAIL ' + retri_file[1][1:] + '\n'
                continue
                   
            r_cent = test_loader.dataset.frame_ctr[r_fmid[0]][r_fmid[1]][1]
            dist = np.linalg.norm(q_cent - r_cent)
            if dist < dist_thred:
                log_strings += ': SUCCESS ' + retri_file[1][1:] + ' ' + str(dist) + ' \n'
                
                for fill in range(k, 3):
                    one_result.append(1)
                break
            else:
                log_strings += ': FAIL ' + retri_file[1][1:] + ' ' + str(dist) + ' \n'
                one_result.append(0)
        # print(test_loader.dataset.files[q_fmid[0]][q_fmid[1]], q_fmid, retri_file) 
        
        query_lists.append([test_loader.dataset.files[q_fmid[0]][q_fmid[1]], test_loader.dataset.files[top1_fmid[0]][top1_fmid[1]], q_camera_ctr, roll, pitch, yaw])  
        
        # query_lists.append([test_loader.dataset.files[q_fmid[0]][q_fmid[1]], test_loader.dataset.files[top1_fmid[0]][top1_fmid[1]]])
        eval_results.append(np.array(one_result))

        # print('current pcd finished in {:.4f}s'.format(time.time() - tt))
    print('Done in {:.1f}s\n'.format(time.time() - t))
    eval_results = np.array(eval_results)
    
    ####################################################################
    # if not os.path.exists('eval/indoor'):
    #     os.makedirs('eval/indoor')
    # scene_sub_info=f"eval/indoor/{args.eval_mode}_{args.backbone}_{args.groups_num}.csv"
    # query_results = np.sum(eval_results, axis=1)
    # for i in range(len(eval_results)):
    #     if query_results[i]>0:
    #         query_results[i] = 1 
    #     query_file, retri_file, camera_ctr, roll, pitch, yaw = query_lists[i]
    #     query_file = query_file.split('input_pcd_0mean')
    #     retri_file = retri_file.split('input_pcd_0mean')
    #     line=[query_file[1], retri_file[1], eval_results[i][0], query_results[i], camera_ctr[0], camera_ctr[1], camera_ctr[2], roll, pitch, yaw] 
        
    #     # line=[query_file[0],query_file[1], eval_results[i][0], query_results[i], retri_file[0],retri_file[1]] 
              
    #     with open(scene_sub_info, 'a', newline='') as csvfile:
    #                 posewriter = csv.writer(
    #                     csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #                 posewriter.writerow(line)
   
    ####################################################################
    
    num_test = eval_results.shape[0]
    accu_results = np.sum(eval_results, axis=0)
    print('Evaluation Results',
            '\n    with', len(batchInd_fileId), 'stored pcd', num_test, 'test pcd',
            '\n    with distance threshold', dist_thred)
    
    db_string = 'Database contains ' + str(len(batchInd_fileId)) + ' point clouds\n'
    qr_string = 'Total number of point cloud tested: ' + str(num_test) + '\n'
    thre_string = 'With distance threshold' + str(dist_thred) + '\n'
    result_strings = ''
    print(accu_results)
    for k, accum1 in enumerate(accu_results):
        result_string = ' - Top ' + str(k+1) + ' recall = ' + str(accum1/num_test)
        print(result_string)
        result_strings += (result_string + '\n')
    
    



if __name__ == "__main__":
    evaluate()