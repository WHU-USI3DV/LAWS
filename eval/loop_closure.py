import os
import sys
import torch
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from eval.eval_sequence import *
from models.model_factory import model_factory

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")


def evaluate_checkpoint(model, save_path, cfg):
    # try:
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    
    # try:
    #   model.load_state_dict(checkpoint['model_state_dict'])
    #   print('Use pretrain model')
    # except:
    #   model.load_state_dict(checkpoint)
    #   print('Retrieval model')
    
    model.load_state_dict(checkpoint['state_dict'])
    print('Use pretrain model')
      
    model = model.cuda()
    model.eval()
    return evaluate_sequence_reg(model, cfg)


def evaluate_checkpoint_laws(model, save_path, cfg):
    # try:
    print('Use pretrain model')
    state = torch.load(save_path)  # ,map_location='cuda:0')
    epoch = state['epoch'] + 1
    #### Updating 
    for i in range(epoch):
      if i < cfg.groups_num:
          model.update_aggregators()
    print(model)
    #### Loading
    model.load_state_dict(state['model_state_dict'])
    model = model.cuda()
    model.eval()
    return evaluate_sequence_reg(model, cfg)


if __name__ == "__main__":

    from eval.eval_config import get_config_eval
    cfg = get_config_eval()
    # Get model
    model = model_factory(cfg)  
    model = model.to("cuda").eval()
    # model = DetectAndVLAD.get_model(args=cfg, radius = cfg.radius, num_samples =cfg.num_samples, feature_dim=cfg.local_dim, 
    #                             num_clusters=cfg.num_clusters, batch_size = cfg.eval_batch_size)  
    save_path = cfg.checkpoint_name
    print('Loading checkpoint from: ', save_path)
    logging.info('\n' + ' '.join([sys.executable] + sys.argv))

    if cfg.eval_mode=='laws':
        eval_F1_max = evaluate_checkpoint_laws(model, save_path, cfg)
    elif cfg.eval_mode=='cp':
        eval_F1_max = evaluate_checkpoint(model, save_path, cfg)
    
    
    
    logging.info(
        '\n' + '******************* Evaluation Complete *******************')
    logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
    if 'Kitti' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.kitti_eval_seq))
    elif 'MulRan' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.mulran_eval_seq))
    logging.info('F1 Max: ' + str(eval_F1_max))
