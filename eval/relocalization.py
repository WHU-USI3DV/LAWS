import argparse
import math, torch
import numpy as np
from tqdm import tqdm
import socket
import importlib
import os
import sys
import time 
import logging, pickle
from sklearn.neighbors import KDTree
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from util import parser
import importlib
import MinkowskiEngine as ME
sys.path.append('model')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from models.model_factory import model_factory
from models.MinkLoc3dv2.mink_params import CartesianQuantizer


args = parser.parse_arguments()



BATCH_NUM_QUERIES = args.batch_size
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
# RESULTS_FOLDER =  "save2/"
# if not os.path.exists(RESULTS_FOLDER): os.mkdir(RESULTS_FOLDER)

BASE_PATH='/home/xy/xy/code/Oxford/data/benchmark_datasets'

# DATABASE_FILE = '../FGSN_part1/data/benchmark_datasets/oxford_evaluation_database.pickle'
# QUERY_FILE = '../FGSN_part1/data/benchmark_datasets/oxford_evaluation_query.pickle'

if args.eval_sequence == 'uni':
    DATABASE_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/university_evaluation_database.pickle'
    QUERY_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/university_evaluation_query.pickle'
elif args.eval_sequence == 'res':
    DATABASE_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/residential_evaluation_database.pickle'
    QUERY_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/residential_evaluation_query.pickle'
elif args.eval_sequence == 'bus':
    DATABASE_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/business_evaluation_database.pickle'
    QUERY_FILE = '/home/xy/xy/code/LAWS/misc/pickle_data/business_evaluation_query.pickle'
elif args.eval_sequence == 'oxf':
    DATABASE_FILE = "/home/xy/xy/code/LAWS/misc/pickle_data/oxford_evaluation_database.pickle"
    QUERY_FILE =  "/home/xy/xy/code/LAWS/misc/pickle_data/oxford_evaluation_query.pickle"
    

print('===============================')
print('Evaluate the %s part'%(args.eval_sequence))

def get_sets_dict(filename):
    #[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
    with open(filename, 'rb') as handle:
        trajectories = pickle.load(handle)
        return trajectories
    

DATABASE_SETS= get_sets_dict(DATABASE_FILE)
QUERY_SETS= get_sets_dict(QUERY_FILE)

global DATABASE_VECTORS
DATABASE_VECTORS=[]

global QUERY_VECTORS
QUERY_VECTORS=[]

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def evaluate():
    DATABASE_VECTORS=[]
    QUERY_VECTORS=[]
    #定义device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)
    print('===============================')

    print('Evaluate the %s part'%(args.eval_sequence))
    ''' === Load Model and Backup Scripts === '''
    logging.info('===> Loading weights')
    print('===> Loading weights', args.save_dir, args.evaluate_model)
    state = torch.load(os.path.join('logs', args.save_dir, args.evaluate_model))  # load pth
    try:
      epoch = state['epoch'] + 1
    except:
      epoch = 0
    #### Model
    net = model_factory(args) 
    #### Updating 
    for i in range(epoch):
      if i < args.groups_num:
          net.update_aggregators()
       
    net.load_state_dict(state['model_state_dict'])
    # net.load_state_dict(state)
    net = net.to('cuda')
    net.eval()
    start= time.time()
    # EVALUATE_QUERIES
    # 计算recall
    recall= np.zeros(25)
    count=0
    similarity=[]
    one_percent_recall=[]
    
    
    get_latent_vectors = LatentVectorsFactory(args.backbone, 0.01)
    
    for i in tqdm(range(len(DATABASE_SETS))):
        DATABASE_VECTORS.append(get_latent_vectors(net, DATABASE_SETS[i]))
        
    for j in tqdm(range(len(QUERY_SETS))):
        QUERY_VECTORS.append(get_latent_vectors(net, QUERY_SETS[j]))
    
    print(DATABASE_VECTORS[0].shape)
    
    for m in range(len(QUERY_SETS)):
        for n in range(len(QUERY_SETS)):
            if(m==n):
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS)
            recall+=np.array(pair_recall)
            count+=1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    print('-----------EVALUATE_QUERIES --------')
    ave_recall=recall/count
    print(ave_recall)

    average_similarity= np.mean(similarity)
    print(average_similarity)

    ave_one_percent_recall= np.mean(one_percent_recall)
    print(ave_one_percent_recall)

    end = time.time()
    dur = end-start
    print("testing time %f s"%(dur))
    # with open(output_file, "w") as output:
    #     output.write("Average Recall @N:\n")
    #     output.write(str(ave_recall))
    #     output.write("\n\n")
    #     output.write("Average Similarity:\n")
    #     output.write(str(average_similarity))
    #     output.write("\n\n")
    #     output.write("Average Top 1% Recall:\n")
    #     output.write(str(ave_one_percent_recall))

    return ave_recall[0], ave_one_percent_recall    


def load_pc_file(filename):
    #returns Nx3 matrix

    #输出路径
    pc=np.fromfile(os.path.join(BASE_PATH,filename), dtype=np.float64)
    
    if(pc.shape[0]!= 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc=np.reshape(pc,(pc.shape[0]//3,3))
    return pc


def load_pc_files(filenames):
    pcs=[]
    for filename in filenames:
        pc=load_pc_file(filename)
        if(pc.shape[0]!=4096):
            continue
        pcs.append(pc)
    pcs=np.array(pcs)
    return pcs


def get_latent_vectors(model, dict_to_process):

    model.eval()
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num= BATCH_NUM_QUERIES  
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).type(torch.FloatTensor).to('cuda')
            out = model(feed_tensor,is_training=is_training)
            
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(dict_to_process.keys()):
        file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).type(torch.FloatTensor).to('cuda')
            o1 = model(feed_tensor,is_training=is_training)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    # print(q_output.shape)
    return q_output


def get_recall(m, n, DATABASE_VECTORS, QUERY_VECTORS):

    database_output = DATABASE_VECTORS[m]
    queries_output = QUERY_VECTORS[n]

    # print(len(queries_output))
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        true_neighbors = QUERY_SETS[n][i][m]
        if(len(true_neighbors) == 0):
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(
            np.array([queries_output[i]]),k=num_neighbors)
        
        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if(j == 0):
                    similarity = np.dot(
                        queries_output[i], database_output[indices[0][j]])
                    # similarity = distances[0][j]
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    # print(recall)
    # print(np.mean(top1_similarity_score))
    # print(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall

def evaluate_all_checkpoints():
    # 统计文件夹所有文件：
    
    path=os.path.join('/home/xy/xy/CosPlace-main','logs' ,args.save_dir)
    dirs = os.listdir(path)
    max_recall, max_top1 = 0, 0
    for dir in dirs:
        if '.pth' in dir:
            args.backbone='3d'
            args.num_clusters = 512
            args.radius = 0.1 
            args.num_samples = 128
            args.local_dim = 256 
            args.evalute_model = dir 
            top1_recall, ave_one_percent_recall = evaluate()
            if(max_recall<ave_one_percent_recall):
                max_recall=ave_one_percent_recall
                best_record = dir
            if(max_top1<top1_recall):
                max_top1=top1_recall
            DATABASE_VECTORS=[]
            QUERY_VECTORS=[]

    print(f"The best recall comes from {best_record} ranks {max_recall} at top 1/100 and ranks {max_top1} at top1")     


def evaluate_all_checkpoints():
    # 统计文件夹所有文件：
    
    path=os.path.join('logs' ,args.save_dir)
    dirs = os.listdir(path)
    max_recall, max_top1 = 0, 0
    ckps, epochs = [],[]
    for dir in dirs:
        if '.pth' in dir:
            ckps.append(dir)
            epoch = dir[:-4]
            epoch = int(epoch[17:])
            epochs.append(epoch)
    epochs = np.array(epochs)  
    idx = np.argsort(epochs)
    epochs = epochs[idx]
    dirs = [ckps[i] for i in idx]
    
    # dirs = dirs[:8]
    print(dirs)
    # exit()
    for dir in dirs:
        args.evaluate_model = dir 
        top1_recall, ave_one_percent_recall = evaluate()

    # print(f"The best recall comes from {best_record} ranks {max_recall} at top 1/100 and ranks {max_top1} at top1")     
      
class LatentVectorsFactory:
    def __init__(self, collation_type='default', quantization_step = 0.01):
        self.quantization_step = quantization_step
        if 'mink' in collation_type:
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
            self.collation_fn = self.get_minkloc_latent_vectors
        else:
            self.collation_fn = self.get_latent_vectors
    
      
    def make_sparse_tensor(self, clouds, quantizer):
      coords = [quantizer(e)[0] for e in clouds]
      coords = ME.utils.batched_coordinates(coords)
      # Assign a dummy feature equal to 1 to each point
      feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
      batch = {'coords': coords, 'features': feats}
      return batch
    
    def __call__(self, model, dict_to_process):
        return self.collation_fn(model, dict_to_process)

    def get_latent_vectors(self, model, dict_to_process):

        model.eval()
        is_training = False
        train_file_idxs = np.arange(0, len(dict_to_process.keys()))
        # print(model.mlps2_1[0][0].weight[0])
        batch_num= BATCH_NUM_QUERIES  
        q_output = []
        for q_index in range(len(train_file_idxs)//batch_num):
            file_indices = train_file_idxs[q_index *
                                          batch_num:(q_index+1)*(batch_num)]
            file_names = []
            for index in file_indices:
                file_names.append(dict_to_process[index]["query"])
            queries = load_pc_files(file_names)

            with torch.no_grad():
                feed_tensor = torch.from_numpy(queries).type(torch.FloatTensor).to('cuda')
                
                out = model(feed_tensor,is_training=False)
                
            
            out = out.detach().cpu().numpy()
            out = np.squeeze(out)

            #out = np.vstack((o1, o2, o3, o4))
            q_output.append(out)
        
        q_output = np.array(q_output)
        if(len(q_output) != 0):
            q_output = q_output.reshape(-1, q_output.shape[-1])

        # handle edge case
        index_edge = len(train_file_idxs) // batch_num * batch_num
        if index_edge < len(dict_to_process.keys()):
            file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
            file_names = []
            for index in file_indices:
                file_names.append(dict_to_process[index]["query"])
            queries = load_pc_files(file_names)

            with torch.no_grad():
                feed_tensor = torch.from_numpy(queries).type(torch.FloatTensor).to('cuda')
                o1 = model(feed_tensor,is_training=False)

            output = o1.detach().cpu().numpy()
            output = np.squeeze(output)
            if (q_output.shape[0] != 0):
                q_output = np.vstack((q_output, output))
            else:
                q_output = output

        # print(q_output.shape)
        return q_output
        
    def get_minkloc_latent_vectors(self, model, dict_to_process):
        model.eval()
        train_file_idxs = np.arange(0, len(dict_to_process.keys()))
        # print(model.mlps2_1[0][0].weight[0])
        batch_num= BATCH_NUM_QUERIES  
        q_output = []
        
        self.quantization_step = 0.01
        
        for q_index in range(len(train_file_idxs)//batch_num):
            file_indices = train_file_idxs[q_index *
                                          batch_num:(q_index+1)*(batch_num)]
            file_names = []
            for index in file_indices:
                file_names.append(dict_to_process[index]["query"])
            queries = load_pc_files(file_names)

            with torch.no_grad():
                batch = self.make_sparse_tensor(queries, self.quantizer)
                feed_tensor = {e: batch[e].to('cuda') for e in batch}
                out = model(feed_tensor, is_training=False)
                
            out = out.detach().cpu().numpy()
            out = np.squeeze(out)

            #out = np.vstack((o1, o2, o3, o4))
            q_output.append(out)
        
        q_output = np.array(q_output)
        if(len(q_output) != 0):
            q_output = q_output.reshape(-1, q_output.shape[-1])

        # handle edge case
        index_edge = len(train_file_idxs) // batch_num * batch_num
        if index_edge < len(dict_to_process.keys()):
            file_indices = train_file_idxs[index_edge:len(dict_to_process.keys())]
            file_names = []
            for index in file_indices:
                file_names.append(dict_to_process[index]["query"])
            queries = load_pc_files(file_names)

            with torch.no_grad():
                batch = self.make_sparse_tensor(queries, self.quantizer)
                feed_tensor = {e: batch[e].to('cuda') for e in batch}
                o1 = model(feed_tensor,is_training=False)

            output = o1.detach().cpu().numpy()
            output = np.squeeze(output)
            if (q_output.shape[0] != 0):
                q_output = np.vstack((q_output, output))
            else:
                q_output = output

        # print(q_output.shape)
        return q_output
      
if __name__ == "__main__":
    
    # evaluate_all_checkpoints()
    evaluate()