# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.MinkLoc3dv2.layers.netvlad import NetVLADLoupe
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor

class GAP(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GAP, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()
        self.eps = eps
    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        
        temp = ME.SparseTensor(x.F.clamp(min=self.eps),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        
        return temp.F    # Return (batch_size, n_features) tensor


class GeMt(nn.Module):
    def __init__(self, input_dim, p=1, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        
    def gem(self, x, p=3, eps=1e-6):
        # return F.avg_pool2d(x.clamp(min=eps).pow(p)).pow(1./p)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def flatten(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]
      
    def forward(self, x):
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros
        features = features.transpose(2,1).unsqueeze(-1)
        x = self.gem(features, p=self.p, eps=self.eps)
        x = self.flatten(x)
        
        return x
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Avgt(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim
        
    def forward(self, x):
        assert x.F.shape[1] == self.feature_size
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros
        features = features.transpose(2,1)  # (batch_size, feature_size, n_points)
        x = F.avg_pool1d(features, kernel_size= features.size(-1)).squeeze(-1)
        
        return x
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Avg(nn.Module):
    def __init__(self, input_dim, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.feature_size = input_dim
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()
        
        
    def forward(self, x):
        
        assert x.F.shape[1] == self.feature_size
        temp = ME.SparseTensor(x.F.clamp(min=self.eps),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        
        x = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        
        
        # features = x.decomposed_features
        # # features is a list of (n_points, feature_size) tensors with variable number of points
        # features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # # features is (batch_size, n_points, feature_size) tensor padded with zeros
        # features = features.transpose(2,1)  # (batch_size, feature_size, n_points)
        # x = F.avg_pool1d(features, kernel_size= features.size(-1)).squeeze(-1)
        
        return x.F
      
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class GeMLinear(nn.Module):
    def __init__(self, input_dim, out_features, p=3, eps=1e-6):
        super(GeMLinear, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = out_features
        self.p = p
        self.eps = eps
        self.p = nn.Parameter(torch.ones(1) * p)
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        #temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        
        temp = ME.SparseTensor(temp.F.pow(1./self.p),
                               coordinate_manager = x.coordinate_manager,
                               coordinate_map_key = x.coordinate_map_key)
        
        return temp    # Return (batch_size, n_features) tensor


class NetVLADWrapper(nn.Module):
    def __init__(self, feature_size, output_dim, gating=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, cluster_size=64, output_dim=output_dim, gating=gating,
                                     add_batch_norm=True)

    def forward(self, x: ME.SparseTensor):
        # x is (batch_size, C, H, W)
        assert x.F.shape[1] == self.feature_size
        
        features = x.decomposed_features
        # features is a list of (n_points, feature_size) tensors with variable number of points
        batch_size = len(features)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        # features is (batch_size, n_points, feature_size) tensor padded with zeros
        
        x = self.net_vlad(features)
        assert x.shape[0] == batch_size
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor
