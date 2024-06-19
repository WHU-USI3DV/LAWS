# Warsaw University of Technology

import torch.nn as nn
import os
from models.MinkLoc3dv2.minkloc import MinkLoc, MinkLocLAWS
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.MinkLoc3dv2.layers.eca_block import ECABasicBlock
from models.MinkLoc3dv2.minkfpn import MinkFPN
from models.MinkLoc3dv2.layers.pooling_wrapper import PoolingWrapper
import configparser
import torch
import numpy as np
from typing import List
import time
from abc import ABC, abstractmethod
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

class Quantizer(ABC):
    @abstractmethod
    def __call__(self, pc):
        pass

class CartesianQuantizer(Quantizer):
    def __init__(self, quant_step: float):
        self.quant_step = quant_step

    def __call__(self, pc):
        # Converts to polar coordinates and quantizes with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3
        quantized_pc, ndx = ME.utils.sparse_quantize(pc, quantization_size=self.quant_step, return_index=True)
        # Return quantized coordinates and index of selected elements
        return quantized_pc, ndx


class PolarQuantizer(Quantizer):
    def __init__(self, quant_step: List[float]):
        assert len(quant_step) == 3, '3 quantization steps expected: for sector (in degrees), ring and z-coordinate (in meters)'
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)
        self.theta_range = int(360. // self.quant_step[0])
        self.quant_step = torch.tensor(quant_step, dtype=torch.float)

    def __call__(self, pc):
        # Convert to polar coordinates and quantize with different step size for each coordinate
        # pc: (N, 3) point cloud with Cartesian coordinates (X, Y, Z)
        assert pc.shape[1] == 3

        # theta is an angle in degrees in 0..360 range
        theta = 180. + torch.atan2(pc[:, 1], pc[:, 0]) * 180./np.pi
        # dist is a distance from a coordinate origin
        dist = torch.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
        z = pc[:, 2]
        polar_pc = torch.stack([theta, dist, z], dim=1)
        # Scale each coordinate so after quantization with step 1. we got the required quantization step in each dim
        polar_pc = polar_pc / self.quant_step
        quantized_polar_pc, ndx = ME.utils.sparse_quantize(polar_pc, quantization_size=1., return_index=True)
        # Return quantized coordinates and indices of selected elements
        return quantized_polar_pc, ndx

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

class MinkLocParams(object):
    def __init__(self):
        self.model=MinkLoc
        self.planes=64,128,64,32
        self.layers=1,1,1,1
        self.num_top_down=2
        self.conv0_kernel_size=5
        self.feature_size=256
        self.block='ECABasicBlock'
        self.pooling='netvlad'

        self.coordinates='cartesian'
        self.quantization_step=0.01

        self.normalize_embeddings=False
        self.output_dim = 256

def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == 'BasicBlock':
        block_module = BasicBlock
    elif block_name == 'Bottleneck':
        block_module = Bottleneck
    elif block_name == 'ECABasicBlock':
        block_module = ECABasicBlock
    else:
        raise NotImplementedError('Unsupported network block: {}'.format(block_name))

    return block_module

class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']
        
        self.model_params_path = model_params_path
        self.model = params.get('model')
        print(self.model)
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        #######################################################################
        # Model dependent
        #######################################################################

        self.coordinates = params.get('coordinates', 'polar')
        assert self.coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.coordinates}'

        if 'polar' in self.coordinates:
            # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
            self.quantization_step = tuple([float(e) for e in params['quantization_step'].split(',')])
            assert len(self.quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
            self.quantizer = PolarQuantizer(quant_step=self.quantization_step)
        elif 'cartesian' in self.coordinates:
            # Single quantization step for cartesian coordinates
            self.quantization_step = params.getfloat('quantization_step')
            self.quantizer = CartesianQuantizer(quant_step=self.quantization_step)
        else:
            raise NotImplementedError(f"Unsupported coordinates: {self.coordinates}")

        # Use cosine similarity instead of Euclidean distance
        # When Euclidean distance is used, embedding normalization is optional
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)

        # Size of the local features from backbone network (only for MinkNet based models)
        self.feature_size = params.getint('feature_size', 256)
        if 'planes' in params:
            self.planes = tuple([int(e) for e in params['planes'].split(',')])
        else:
            self.planes = tuple([32, 64, 64])

        if 'layers' in params:
            self.layers = tuple([int(e) for e in params['layers'].split(',')])
        else:
            self.layers = tuple([1, 1, 1])

        self.num_top_down = params.getint('num_top_down', 1)
        self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)
        self.block = params.get('block', 'BasicBlock')
        self.pooling = params.get('pooling', 'GeM')

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')

class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)          # Model saving frequency (in epochs)
        self.num_workers = params.getint('num_workers', 0)

        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)
        # When batch_split_size is non-zero, multistage backpropagation is enabled
        self.batch_split_size = params.getint('batch_split_size', None)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs+1]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        elif self.loss == 'truncatedsmoothap':
            # Number of best positives (closest to the query) to consider
            self.positives_per_query = params.getint("positives_per_query", 4)
            # Temperatures (annealing parameter) and numbers of nearest neighbours to consider
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)    # Margin used in loss function

        # Similarity measure: based on cosine similarity or Euclidean distance
        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)
        self.set_aug_mode = params.getint('set_aug_mode', 1)    # Augmentation mode (1 is default)
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')

def model_factory(model_params: ModelParams):
        in_channels = 1

        if model_params.model == 'mink':
            block_module = create_resnet_block(model_params.block)
            backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                              num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                              block=block_module, layers=model_params.layers, planes=model_params.planes)
            pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
                                    output_dim=model_params.output_dim)
            model = MinkLoc(backbone=backbone, pooling=pooling, normalize_embeddings=model_params.normalize_embeddings)
        elif model_params.model == 'mink_laws':
            block_module = create_resnet_block(model_params.block)
            backbone = MinkFPN(in_channels=in_channels, out_channels=model_params.feature_size,
                              num_top_down=model_params.num_top_down, conv0_kernel_size=model_params.conv0_kernel_size,
                              block=block_module, layers=model_params.layers, planes=model_params.planes)
            # pooling = PoolingWrapper(pool_method=model_params.pooling, in_dim=model_params.feature_size,
            #                         output_dim=model_params.output_dim)
            model = MinkLocLAWS(backbone=backbone, 
                                pool_method = model_params.pooling,
                                in_dim=model_params.feature_size,
                                output_dim=model_params.output_dim,
                                normalize_embeddings=model_params.normalize_embeddings)
        elif model_params.model == 'ptc':
            pooling = PoolingWrapperPTC(pool_method=model_params.pooling, 
                                        in_dim=model_params.feature_size,
                                        output_dim=model_params.output_dim)
            model = PTC_Net(quantization_step=0.01, pooling=pooling)
        elif model_params.model == 'ptc_laws':
            model = PTC_Net_LAWS(model_params, quantization_step=0.01)  
        else:
            raise NotImplementedError('Model not implemented: {}'.format(model_params.model))

        return model
      

def make_collate_fn(quantizer, batch_split_size=128):
  def collate_fn(data_list):
        # Constructs a batch object
        
        clouds = [e[0] for e in data_list]  #(B,S,4096,3)
        labels = [e[1] for e in data_list]  #(B)
        
        clouds, labels = torch.from_numpy(np.array(clouds)), torch.tensor(labels)
        
        S = clouds.shape[1]
        
        clouds=torch.flatten(clouds, start_dim=0, end_dim=1)
        labels = torch.flatten(labels.unsqueeze(-1).repeat(1,S))
       
        # Compute positives and negatives mask
        positives_mask = [[torch.eq(anc, e)  for e in labels]for anc in labels]
        negatives_mask = [[not torch.eq(anc, e)  for e in labels]for anc in labels]
        # positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        # negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)
        for i in range(len(positives_mask)):
          for j in range(len(positives_mask)):
            if i==j:
              positives_mask[i][j]=False
        
        coords = [quantizer(e)[0] for e in clouds]

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            
            batch = {'coords': coords, 'features': feats, 'batch': clouds.type(torch.float32)}

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                batch_temp = clouds[i:i + batch_split_size]
                minibatch = {'coords': c, 'features': f, 'batch': batch_temp}
                batch.append(minibatch)
                
        return batch, labels, positives_mask, negatives_mask
  return collate_fn


def make_sparse_tensor(clouds, quantizer, quantization_step):
      coords = [quantizer(e)[0] for e in clouds]
      coords = ME.utils.batched_coordinates(coords)
      # Assign a dummy feature equal to 1 to each point
      feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
      batch = {'coords': coords, 'features': feats, 'batch': clouds.type(torch.float32)}
      return batch  

# v2
def make_sparse_tensor_rgb(clouds, quantizer, quantization_step):
    
    clouds = clouds.squeeze(1)
    batch_size, num_points = clouds.shape[0], clouds.shape[1]
    coords, feats, labels = [], [], []
    for cloud in clouds:
      xyz, rgb = cloud[:,:3].float(), cloud[:,3:].float()
      label = torch.ones((num_points, 1),dtype=torch.int32)
      # Quantize the input
      discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
          coordinates=xyz,
          features=rgb,
          labels=label,
          quantization_size=quantization_step,
          ignore_label=-100)
      coords.append(discrete_coords)
      feats.append(unique_feats)
      labels.append(unique_labels)
    coords_batch, feats_batch, labels_batch = [], [], []
    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)
    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))  
    batch = {'coords': coords_batch, 'features': feats_batch}
    return batch
    
# def make_sparse_tensor_rgb(clouds, quantizer):
#       clouds = clouds.squeeze(1)
#       batch_xyz, batch_rgb = clouds[:,:,:3].float(), clouds[:,:,3:].float()
#       coords, feats = [], []
#       for xyz in batch_xyz:
#         coords.append(xyz)
#       for rgb in batch_rgb:
#         feats.append(rgb)  
#       coords, feats = ME.utils.sparse_collate(coords, feats)
  
#       batch = {'coords': coords, 'features': feats, 'batch': batch_xyz.type(torch.float32)}
#       return batch
    
# def make_sparse_tensor_rgb(clouds, quantizer):
      
#       clouds = clouds.squeeze(1)
#       batch_xyz, batch_rgb = clouds[:,:,:3].float(), clouds[:,:,3:].float()
#       xyz =[]
#       for i in range(batch_xyz.shape[0]):
#           xyz.append(batch_xyz[i,:,:])
      
#       coords = [quantizer(e)[0] for e in xyz]
#       coords = ME.utils.batched_coordinates(coords)
#       # Assign a dummy feature equal to 1 to each point
#       feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
#       batch = {'coords': coords, 'features': feats, 'batch': clouds.type(torch.float32)}
#       return batch 
    
        