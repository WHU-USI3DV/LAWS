from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2).contiguous()
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, use_rgb=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.input_dim = 6 if use_rgb else 3
        self.conv1 = torch.nn.Conv2d(1, 64, (1, self.input_dim))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        # x [B N 3]  or [B N 6]
        if x.shape[-1] == 3:
          x=x.unsqueeze(1)  # x [B 1 N 3]
          batchsize = x.size()[0]
          trans = self.stn(x)
          x = torch.matmul(torch.squeeze(x), trans)
          x = x.view(batchsize, 1, -1, 3)
        elif x.shape[-1] == 6:
          xyz = x[:,:,:3]
          rgb = x[:,:,3:]
          
          xyz, rgb = xyz.unsqueeze(1), rgb.unsqueeze(1)
          batchsize = xyz.size()[0]
          trans = self.stn(xyz)
          xyz = torch.matmul(torch.squeeze(xyz), trans)
          xyz = xyz.view(batchsize, 1, -1, 3)
          x = torch.cat((xyz, rgb), dim = -1)
        
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetVlad(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024, use_rgb = False):
        super(PointNetVlad, self).__init__()
        
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool, use_rgb=use_rgb)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x, is_training= True):
        # x [B N 3]
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x

class PointNetVladLAWS(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetVladLAWS, self).__init__()
        self.curr_group = 0
        self.num_points = num_points
        self.output_dim = output_dim

        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        self.buffer_bn = nn.BatchNorm2d(1024)
        self.bufferlayers = nn.ModuleList() 
        self.aggregators = nn.ModuleList()                              
        # self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
        #                              output_dim=output_dim, gating=True, add_batch_norm=True,
        #                              is_training=True)
    
    def update_aggregators(self):
        conv_buffer = torch.nn.Conv2d(1024, 1024, (1, 1))
        aggregator = NetVLADLoupe(feature_size=1024, max_samples=self.num_points, cluster_size=64,
                                     output_dim=self.output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)
        self.bufferlayers.append(conv_buffer)
        self.aggregators.append(aggregator)    

    def feature_extraction_training_module(self, x):
        x = self.buffer_bn(self.bufferlayers[self.curr_group](x))
        x = self.aggregators[self.curr_group](x)  
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def feature_extraction_inference_module(self, x):
        global_features = []
        for (buffer, aggregator) in zip(self.bufferlayers, self.aggregators):
            x = self.buffer_bn(buffer(x))
            des = aggregator(x)  
            des = F.normalize(des, p=2, dim=1)
            global_features.append(des)
        global_features = torch.cat(global_features, 1)   
        return global_features

    def forward(self, x, is_training= True):
        # x [B N 3] or [B N 6]
        x = self.point_net(x)
        if is_training:
          x = self.feature_extraction_training_module(x)
        else:
          x = self.feature_extraction_inference_module(x)
        
        return x

# 无缓冲层的版本
class PointNetVladLAWScopy(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024, use_rgb = False):
        super(PointNetVladLAWScopy, self).__init__()
        self.curr_group = 0
        self.num_points = num_points
        self.output_dim = output_dim

        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool, use_rgb=use_rgb)

        self.aggregators = nn.ModuleList()                              
        # self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
        #                              output_dim=output_dim, gating=True, add_batch_norm=True,
        #                              is_training=True)
    
    def update_aggregators(self):
        aggregator = NetVLADLoupe(feature_size=1024, max_samples=self.num_points, cluster_size=64,
                                     output_dim=self.output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)
        self.aggregators.append(aggregator)    

    def feature_extraction_training_module(self, x):
        x = self.aggregators[self.curr_group](x)  
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def feature_extraction_inference_module(self, x):
        
        global_features = []
        for aggregator in self.aggregators:
            des = aggregator(x)  
            des = F.normalize(des, p=2, dim=1)
            global_features.append(des)
        global_features = torch.cat(global_features, 1)   
        return global_features

    def forward(self, x, is_training= True):
        # x [B N 3] or [B N 6]
        
        x = self.point_net(x)
        if is_training:
          x = self.feature_extraction_training_module(x)
        else:
          x = self.feature_extraction_inference_module(x)
        
        return x


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos
  
def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False, soft_margin=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies)** 2).sum(2)
    if soft_margin:
        loss = loss.clamp(max=88)
        loss = torch.log(1 + torch.exp(loss))   # softplus
    else:
        loss = loss.clamp(min=0.0)              # hinge  function
    if lazy:                                    # lazy = true
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.mean(1)
    if ignore_zero_loss:                        # false
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies)** 2).sum(2)
    if soft_margin:
        second_loss = second_loss.clamp(max=88)
        second_loss = torch.log(1 + torch.exp(second_loss))
    else:
        second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.mean(1)
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    
    return total_loss


if __name__ == '__main__':
    num_points = 4096
    sim_data = Variable(torch.rand(44, 3, num_points))
    sim_data = sim_data.cuda()

    pnv = PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    
    pnv.update_aggregators(0,False)
    pnv.train()
    pnv.to('cuda')
    out3 = pnv(sim_data)
    print('pnv', out3.size())
