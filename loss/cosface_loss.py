
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math

class Circle_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.2, gamma=512):
      super().__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.m = m 
      self.gamma = gamma
      self.fc_layer = nn.Linear(in_features, out_features)
      self.soft_plus = nn.Softplus()
      
    def convert_label_to_similarity(self, normed_feature, label):
      # normed_feature[batch, n_class], label[batch]
      similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
      label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

      positive_matrix = label_matrix.triu(diagonal=1)
      negative_matrix = label_matrix.logical_not().triu(diagonal=1)

      similarity_matrix = similarity_matrix.view(-1)
      positive_matrix = positive_matrix.view(-1)
      negative_matrix = negative_matrix.view(-1)
      return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]
    
    def forward(self, descriptor, label):
      pred = self.fc_layer(descriptor)
      pred = F.normalize(pred, p=2, dim=1)  # L2 normalize
      
      sp, sn = self.convert_label_to_similarity(pred, label)
      
      ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
      an = torch.clamp_min(sn.detach() + self.m, min=0.)

      delta_p = 1 - self.m
      delta_n = self.m

      logit_p = - ap * (sp - delta_p) * self.gamma
      logit_n = an * (sn - delta_n) * self.gamma

      loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
      
      return loss


class Circle_Loss2(nn.Module):
    def __init__(self, m=0.2, gamma=512):
      super().__init__()
      self.m = m 
      self.gamma = gamma
      self.soft_plus = nn.Softplus()
      
    def convert_label_to_similarity(self, normed_feature, label):
      # normed_feature[batch, n_class], label[batch]
      similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
      label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

      positive_matrix = label_matrix.triu(diagonal=1)
      negative_matrix = label_matrix.logical_not().triu(diagonal=1)

      similarity_matrix = similarity_matrix.view(-1)
      positive_matrix = positive_matrix.view(-1)
      negative_matrix = negative_matrix.view(-1)
      return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]
    
    def forward(self, descriptor, label):
    #   pred = self.fc_layer(descriptor)
    #   pred = F.normalize(pred, p=2, dim=1)  # L2 normalize
      
      sp, sn = self.convert_label_to_similarity(descriptor, label)
      
      ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
      an = torch.clamp_min(sn.detach() + self.m, min=0.)

      delta_p = 1 - self.m
      delta_n = self.m

      logit_p = - ap * (sp - delta_p) * self.gamma
      logit_n = an * (sn - delta_n) * self.gamma

      loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
      
      return loss

class Circle_Loss_Euclidean_Distance(nn.Module):
    def __init__(self, m=0.2, gamma=256):
      super(Circle_Loss_Euclidean_Distance, self).__init__()
      self.m = m 
      self.gamma = gamma
      self.soft_plus = nn.Softplus()
      
    def convert_label_to_similarity(self, normed_feature, label):
      # normed_feature[batch, n_class], label[batch]
    #   similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
      similarity_matrix = self.convert_descriptor_to_distance(normed_feature, squared = False) 
      label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

      positive_matrix = label_matrix.triu(diagonal=1)
      negative_matrix = label_matrix.logical_not().triu(diagonal=1)

      similarity_matrix = similarity_matrix.view(-1)
      positive_matrix = positive_matrix.view(-1)
      negative_matrix = negative_matrix.view(-1)
      
      return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]
          
    def convert_descriptor_to_distance(self, A, squared = False, eps = 1e-8):
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        
        if squared:
            return res
        else:
            res = res.clamp(min = eps).sqrt()
            return res

    def forward(self, descriptor, label):
      # print(descriptor.shape, label.shape)
      
      sp, sn = self.convert_label_to_similarity(descriptor, label)
      
    #   ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
    #   an = torch.clamp_min(sn.detach() + self.m, min=0.)
      ap = torch.clamp_min(sp.detach() + self.m, min=0.)
      an = torch.clamp_min(- sn.detach() + 1 + self.m, min=0.)
      
      delta_p = self.m
      delta_n = 1 - self.m

      logit_p = ap * (sp - delta_p) * self.gamma
      logit_n = - an * (sn - delta_n) * self.gamma
      loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
      
      return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
          
class MultiFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, in_channel, num_class, alpha=0.25, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
        # if self.alpha is None:
        #     self.alpha = torch.ones(self.num_class, 1)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
        #     self.alpha = self.alpha / self.alpha.sum()
        # elif isinstance(self.alpha, float):
        #     alpha = torch.ones(self.num_class, 1)
        #     alpha = alpha * (1 - self.alpha)
            
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
            
        # else:
        #     raise TypeError('Not support alpha type')
        self.alpha = alpha
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, descriptors, input, target):
        batch_size = descriptors.shape[0]//2
        anchors = descriptors[:batch_size]
        positives =  descriptors[-batch_size:]
        sim = torch.tensor([torch.dot(anchors[i], positives[i]) for i in range(batch_size)])
        similarities = torch.cat((sim,sim),dim=0).to("cuda")
         
        logit = F.softmax(input, dim=1)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        # if alpha.device != logit.device:
        #     alpha = alpha.to("cuda")

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
        gamma = self.gamma
        loss = -1 * alpha * torch.pow((1 - similarities), gamma) * logpt
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class ArcMarginModel(nn.Module):
    def __init__(self, emb_size, num_classes, m=0.5,s=64, easy_margin=False):
        super(ArcMarginModel, self).__init__()
 
        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        # num_classes 训练集中总的人脸分类数
        # emb_size 特征向量长度
        nn.init.xavier_uniform_(self.weight)
        # 使用均匀分布来初始化weight
 
        self.easy_margin = easy_margin
        self.m = m
        # 夹角差值 0.5 公式中的m
        self.s = s
        # 半径 64 公式中的s
        # 二者大小都是论文中推荐值
 
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # 差值的cos和sin
        self.th = math.cos(math.pi - self.m)
        # 阈值，避免theta + m >= pi
        self.mm = math.sin(math.pi - self.m) * self.m
 
    def forward(self, input, label):
        # x = F.normalize(input)
        W = F.normalize(self.weight)
        # 正则化
        cosine = F.linear(input, W)
        # cos值
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # sin
        phi = cosine * self.cos_m - sine * self.sin_m
        # cos(theta + m) 余弦公式
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            # 如果使用easy_margin
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        output *= self.s
        
        # 乘以半径
        return output
      
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())  # 16*256  256*204 -> 16*204
    w1 = torch.norm(x1, 2, dim) # 16
    w2 = torch.norm(x2, 2, dim) # 204
    # print(w1,w2)
    # exit()
    # a = ip / torch.ger(w1,w2).clamp(min=eps)
    
    return ip / torch.ger(w1,w2).clamp(min=eps)

class Linear(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc = nn.Linear(in_features, out_features)
        # nn.init.xavier_uniform_(self.fc)
    def forward(self, input):
        output = self.fc(input)
        return output
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
     
class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        one_hot = torch.zeros_like(cosine)
        # print(one_hot.device, label.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m)
        return output
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class MarginCosineProductTest(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, input):
        cosine = cosine_sim(input, self.weight)
        output = self.s * cosine
        return output
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
 
def pairwise_dist(descriptor):
    ''' Computes pairwise distance

    :param A: (B x D) containing descriptors of A  6*2
    :param B: (B x D) containing descriptors of B  6*2
    :return: (B x B) tensor. Element[i,j,k] denotes the distance between the jth descriptor in ith model of A,
             and kth descriptor in ith model of B
    '''
    print(A.shape,B.shape)
    A = B = descriptor
    Na = A.shape[0]
    Nb = B.shape[0]
    A = A.unsqueeze(1).repeat(1,Na,1)  # (B, B, D)
    B = B.unsqueeze(0).repeat(Nb,1,1)  # (B, B, D)
    
    # dist= torch.pow(A-B,2).float().sum(axis=3).sqrt()   # (B, N, N)
    dist= torch.norm(A-B, dim=2)   # (B, N, N)
    # dist= torch.abs(A-B).float().sum(axis=3)   # (B, N, N)
    
    return dist

def best_neg_distance(query, neg_vecs):
    num_neg = neg_vecs.shape[0]
    query_copies = query.repeat(int(num_neg), 1)
    diff = ((neg_vecs - query_copies) ** 2).sum(1)
    
    # # diff = torch.sum(diff, dim=1, keepdim=False)
    # print(type(diff),diff.shape, diff)
    # min_neg = torch.min(diff)
    # max_neg = torch.max(diff)
    
    return diff

def triplet_loss(descriptors, margin, use_min=False, lazy=False, ignore_zero_loss=False, use_more=True):
    batch = descriptors.shape[0]//2
    q_vecs = descriptors[:batch]
    pos_vecs =  descriptors[-batch:]
    diff_pos = ((pos_vecs - q_vecs) ** 2).sum(1)
    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    # if use_min:
    #     positive = min_pos
    # else:
    #     positive = max_pos
    triplet_loss = 0
    for i,q_vec in enumerate(q_vecs):
        neg_vecs = torch.cat((q_vecs[:i],q_vecs[i+1:]),dim=0)
        if use_more:
            neg_vecs = torch.cat((q_vecs[:i],q_vecs[i+1:],pos_vecs[:i],pos_vecs[i+1:]),dim=0)
        else:
            neg_vecs = torch.cat((q_vecs[:i],q_vecs[i+1:]),dim=0)
        # print(i, neg_vecs.shape, q_vec[0], neg_vecs[i,0], q_vecs[i,0], q_vecs[i+1,0])
        q_negs = best_neg_distance(q_vec, neg_vecs)
        q_pos = diff_pos[i].repeat(len(q_negs))
        loss = margin + q_pos - q_negs
        loss = loss.clamp(min=0.0)
        if lazy:
            loss = loss.max(0)[0]
        else:
            loss = loss.sum(0)
        triplet_loss += loss
    triplet_loss /= batch   
    return triplet_loss

    
    