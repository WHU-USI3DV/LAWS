# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from models.MinkLoc3dv2.layers.pooling_wrapper import PoolingWrapper


class MinkLoc(torch.nn.Module):
    def __init__(self, backbone: nn.Module, pooling: PoolingWrapper, normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.normalize_embeddings = normalize_embeddings
    
    def update_aggregators(self):
       print('No update!')
       
    def forward(self, batch, is_training =True):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        
        x = self.backbone(x)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.pooling.in_dim, f'Backbone output tensor has: {x.shape[1]} channels. ' \
                                                  f'Expected: {self.pooling.in_dim}'
        
        x  = self.pooling(x)
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=1)
        
        if hasattr(self.pooling, 'stats'):
            self.stats.update(self.pooling.stats)
        #x = x.flatten(1)
        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        # assert x.shape[1] == self.pooling.output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
        #                                               f'Expected: {self.pooling.output_dim}'
        
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')


class MinkLocLAWS(torch.nn.Module):
    def __init__(self, backbone: nn.Module, 
                 pool_method: str = 'GeM',
                 in_dim: int = 256,
                 output_dim: int = 256,
                 normalize_embeddings: bool = False):
        super().__init__()
        self.backbone = backbone
        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.normalize_embeddings = normalize_embeddings
        self.stats = {}
        self.aggregators = nn.ModuleList()
        self.curr_group = 0
        print('self.normalize_embeddings', self.normalize_embeddings)
    
    def update_aggregators(self):
        pooling = PoolingWrapper(pool_method=self.pool_method, in_dim=self.in_dim,
                                      output_dim=self.output_dim)
        self.aggregators.append(pooling)

    def feature_extraction_training_module(self, x):
        x = self.aggregators[self.curr_group](x) 
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=1)
        return x
      
    def feature_extraction_inference_module(self, x):
        global_features = []
        for aggregator in self.aggregators:
            g_des = aggregator(x)
            if self.normalize_embeddings:
              g_des = F.normalize(g_des, p=2, dim=1)
            global_features.append(g_des)
        global_features = torch.cat(global_features, 1)   
        return global_features
      
    def forward(self, batch, is_training =True):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
       
        x = self.backbone(x)
        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.aggregators[self.curr_group].in_dim, f'Backbone output tensor has: {x.shape[1]} channels. ' \
                                                  f'Expected: {self.pooling.in_dim}'
        
        if is_training:
          x = self.feature_extraction_training_module(x)
        else:
          x = self.feature_extraction_inference_module(x)
        
        if hasattr(self.aggregators[self.curr_group], 'stats'):
            self.stats.update(self.aggregators[self.curr_group].stats)
        #x = x.flatten(1)
        assert x.dim() == 2, f'Expected 2-dimensional tensor (batch_size,output_dim). Got {x.dim()} dimensions.'
        # assert x.shape[1] == self.aggregators[self.curr_group].output_dim, f'Output tensor has: {x.shape[1]} channels. ' \
        #                                               f'Expected: {self.pooling.output_dim}'
        
        return x

    def print_info(self):
        print('Model class: MinkLoc')
        n_params = sum([param.nelement() for param in self.parameters()])
        print(f'Total parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.backbone.parameters()])
        print(f'Backbone: {type(self.backbone).__name__} #parameters: {n_params}')
        n_params = sum([param.nelement() for param in self.pooling.parameters()])
        print(f'Pooling method: {self.pooling.pool_method}   #parameters: {n_params}')
        print('# channels from the backbone: {}'.format(self.pooling.in_dim))
        print('# output channels : {}'.format(self.pooling.output_dim))
        print(f'Embedding normalization: {self.normalize_embeddings}')
