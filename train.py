
import os, sys
import torch
import random
import logging
import numpy as np
from glob import glob
import torchvision.transforms as T
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from collections import defaultdict
from tqdm import tqdm
from datasets.train_datasets import *
from datasets.data_utils import get_sets_dict
from models.model_factory import model_factory
from models.MinkLoc3dv2.mink_params import make_sparse_tensor, make_sparse_tensor_rgb,CartesianQuantizer, PolarQuantizer
from util import util
from loss import cosface_loss
from util import parser
from datetime import datetime
import time

from tensorboardX import SummaryWriter

#####Args
args = parser.parse_arguments()

start_time = datetime.now()

ALL_DATASETS = [TrainDataset, TrainDatasetv2, TrainDatasetv0, ScannetPRDataset, ScannetPRDatasetv2, ScannetPRDatasetv3]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


class BasicTrainer(object):
    def __init__(self, args):
        self.args = args
        self.T = 2
        self.comp_iter = 0
        self.curr_iter = 0
        self.lamda = 100
        self.get_quantizer(args)
        #### Loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.init_lr = args.lr
        #### Input
        if args.dataset in ['ScannetPRDataset', 'ScannetPRDatasetv2']:
            self.make_tensor = make_sparse_tensor_rgb
        else:
            self.make_tensor = make_sparse_tensor
          
        ######TensorBoardX
        logdir = f"logs/{args.save_dir}"
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)
        
        output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        util.make_deterministic(args.seed)
        util.setup_logging(output_folder, console="debug")
        logging.info(f"The outputs are being saved in {output_folder}")
        # save args parameters
        argsDict = args.__dict__
        with open(os.path.join("logs/", args.save_dir, 'setting.txt'), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        
    def train(self):
      self.train_from_initial()
    
    def train_from_initial(self):  
      #### Model
      self.net = model_factory(args)  
      start_epoch_num = 0
      #### Datasets
      Dataset = dataset_str_mapping[args.dataset]
      groups = [Dataset(args, 
                      args.train_set_folder,
                      M=args.M, 
                      N=args.N, 
                      current_group=n, 
                      min_pointclouds_per_class=args.min_images_per_class) for n in range(args.groups_num)]
       
      # Each group has its own classifier, which depends on the number of classes in the group
      # LMCL
      self.classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
      self.classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in self.classifiers]
      
      model_optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
      
      logging.info(f"Using {len(groups)} groups")
      logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
      logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

      #### Train / evaluation loop
      logging.info("Start training ...")
      logging.info(f"There are {len(groups[0])} classes for the first group, " +
                  f"each epoch has {args.iterations_per_epoch} iterations " +
                  f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                  f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

      for epoch_num in range(start_epoch_num, args.epochs_num):
          
          # Select classifier and dataloader according to epoch
          current_group_num = epoch_num % args.groups_num
          
          self.cur_task = epoch_num
          self.net.to(args.device)
          self.net.train()   
          
          self.classifiers[current_group_num].to(args.device)
          util.move_to_device(self.classifiers_optimizers[current_group_num], args.device)
          
          dataloader = util.InfiniteDataLoader(groups[current_group_num], 
                                                  num_workers=args.num_workers,
                                                  batch_size=args.batch_size, shuffle=True,
                                                  pin_memory=(args.device=="cuda"), drop_last=True)

          dataloader_iterator = iter(dataloader) 
          torch.backends.cudnn.enabled = False
         
          if 'mink' in args.backbone:
              self.get_quantizer(args)
              self.train_one_epoch_sparse_tensor(current_group_num, dataloader_iterator, model_optimizer)
          else:
              self.train_one_epoch(current_group_num, dataloader_iterator, model_optimizer)
          
          self.classifiers[current_group_num].cpu()
          util.move_to_device(self.classifiers_optimizers[current_group_num], "cpu")
          if (epoch_num+1)%4 == 0:
            self.save_ckpt(epoch_num)
          
          # scheduler.step()
          print('---------------',model_optimizer.param_groups[0]['lr'])
      logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    def train_one_epoch(self,current_group_num, dataloader_iterator, model_optimizer):
        total_loss = util.AverageMeter()
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=50):
              pointclouds, targets, _ = next(dataloader_iterator)
              
              pointclouds = pointclouds.squeeze(1).type(torch.FloatTensor)  # (batch, 3, npoint)
              pointclouds, targets = pointclouds.to(args.device), targets.to(args.device)
              
              model_optimizer.zero_grad()
              self.classifiers_optimizers[current_group_num].zero_grad()
              
              descriptors = self.net(pointclouds, True)
              output = self.classifiers[current_group_num](descriptors, targets)
              loss = self.criterion(output, targets)  # LCML
              
              loss.backward()
              
              model_optimizer.step()
              self.classifiers_optimizers[current_group_num].step()

              total_loss.update(loss.item(), 1)
              
              if self.curr_iter % args.log_freq == 0 or self.curr_iter == 0:
                  self.writer.add_scalar('training/initial_loss', total_loss.avg, self.curr_iter)
                  self.writer.add_scalar('training/learning_rate', model_optimizer.param_groups[0]['lr'], self.curr_iter)
                  total_loss.reset()
                  torch.cuda.empty_cache()
              self.curr_iter += 1
              del loss,output, pointclouds 

    def train_one_epoch_sparse_tensor(self,current_group_num, dataloader_iterator, model_optimizer):
        total_loss = util.AverageMeter()
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=50):
              
              model_optimizer.zero_grad()
              self.classifiers_optimizers[current_group_num].zero_grad()
              
              pointclouds, targets, _ = next(dataloader_iterator)
              
              targets = targets.to('cuda')
              batch = self.make_tensor(pointclouds, self.quantizer, self.quantization_step)
              batch = {e: batch[e].to('cuda') for e in batch}
              descriptors = self.net(batch)
              output = self.classifiers[current_group_num](descriptors, targets) # LMCL
              loss = self.criterion(output, targets)  # LCML
              
              
              loss.backward()
              
              model_optimizer.step()
              self.classifiers_optimizers[current_group_num].step()

              total_loss.update(loss.item(), 1)
              
              if self.curr_iter % args.log_freq == 0 or self.curr_iter == 0:
                  self.writer.add_scalar('training/initial_loss', total_loss.avg, self.curr_iter)
                  self.writer.add_scalar('training/learning_rate', model_optimizer.param_groups[0]['lr'], self.curr_iter)
                  total_loss.reset()
                  torch.cuda.empty_cache()
              self.curr_iter += 1
              del loss,output, pointclouds 

    def get_quantizer(self, args):
        if 'polar' in args.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = tuple([float(e) for e in args.quantization_step.split(',')])
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
        elif 'cartesian' in args.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = args.quantization_step
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
    
    def save_ckpt(self, epoch_num):
      #### save every epoch
        filename = f"checkpoint_epoch_{epoch_num}.pth"
        checkpoint_file = 'logs/'+ args.save_dir + filename
        state = {
        'epoch': epoch_num,
        "model_state_dict": self.net.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in self.classifiers]}
        torch.save(state, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")


class BackboneExpansion(BasicTrainer):
    def __init__(self, args):
        BasicTrainer.__init__(self,args)
    
    def train(self):  
      #### Model
      if args.resume_model:
        logging.info('Training from saved weights')
        start_epoch_num = self.loading_weights(args)
      else:
        logging.info('Training from scratch')
        self.net = model_factory(args)  
        start_epoch_num = 0
      
      #### Datasets
      assert args.M == 20
      assert args.groups_num == 8
      Dataset = dataset_str_mapping[args.dataset]
      groups = [Dataset(args, 
                      args.train_set_folder,
                      M=args.M, 
                      N=args.N, 
                      current_group=n, 
                      min_pointclouds_per_class=args.min_images_per_class) for n in range(args.groups_num)]
       
      # Each group has its own classifier, which depends on the number of classes in the group
      # LMCL
      self.classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
      self.classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in self.classifiers]
      logging.info(f"Using {len(groups)} groups")
      logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
      logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

      #### Train / evaluation loop
      logging.info("Start training ...")
      logging.info(f"There are {len(groups[0])} classes for the first group, " +
                  f"each epoch has {args.iterations_per_epoch} iterations " +
                  f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                  f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")
      
      for epoch_num in range(start_epoch_num, args.epochs_num):
          # Select classifier and dataloader according to epoch
          current_group_num = epoch_num % args.groups_num
          
          self.cur_task = epoch_num
          if self.cur_task < args.groups_num:
            self.net.update_aggregators()
          
          # if epoch_num < 16:
          #   lr = self.init_lr
          #   model_optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  
          # elif epoch_num < 24 & epoch_num >= 16:
          #   print('Use SGD optimizer')
          #   lr = self.init_lr *0.5
          #   model_optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  
          # else:
          #   lr = self.init_lr *0.1
          #   model_optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  
          model_optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
          logging.info(f"On epoch {epoch_num} Group_{current_group_num} is under training with aggragator_{len(self.net.aggregators)}")  
          
          self.net.curr_group = current_group_num
          self.net.to(args.device)
          self.net.train()    
          
          self.classifiers[current_group_num].to(args.device)
          util.move_to_device(self.classifiers_optimizers[current_group_num], args.device)
          
          dataloader = util.InfiniteDataLoader(groups[current_group_num], 
                                                  num_workers=args.num_workers,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  pin_memory=(args.device=="cuda"), 
                                                  drop_last=True)
          dataloader_iterator = iter(dataloader) 
          torch.backends.cudnn.enabled = False
          if 'mink' in args.backbone:
              self.train_one_epoch_sparse_tensor(current_group_num, dataloader_iterator, model_optimizer)
          else:
              self.train_one_epoch(current_group_num, dataloader_iterator, model_optimizer)
          
          self.classifiers[current_group_num].cpu()
          util.move_to_device(self.classifiers_optimizers[current_group_num], "cpu")
          if (epoch_num+1)%4 == 0:
            self.save_ckpt(epoch_num)
          
      logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    def loading_weights(self, args):
        # try:
        self.net = model_factory(args)  
        save_path = os.path.join('logs', args.save_dir, args.resume_model)
        print('Use pretrain model')
        state = torch.load(save_path)  # ,map_location='cuda:0')
        epoch = state['epoch'] + 1
        #### Updating 
        for i in range(epoch):
          if i < args.groups_num:
              self.net.update_aggregators()
        print(self.net)
        #### Loading
        self.net.load_state_dict(state['model_state_dict'])
        return epoch

def main(): 
    
    ALL_TRAINERS = [BackboneExpansion, BasicTrainer] 
    trainer_str_mapping = {d.__name__: d for d in ALL_TRAINERS}
    trainer = trainer_str_mapping[args.trainer](args)
    trainer.train()

    
if __name__ == "__main__":
    main()