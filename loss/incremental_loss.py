import torch 
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm 

class NoIncLoss:
    def __init__(self):
        pass 

    def adjust_weight(self, epoch):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.tensor(0, dtype = float, device = 'cuda')

class LwF:
    def __init__(self):
        self.weight = 1000
        self.temperature = 2 
    
    def adjust_weight(self, epoch):
        pass 

    def __call__(self, old_rep, new_rep):
        log_p = torch.log_softmax(new_rep / self.temperature, dim=1)
        q = torch.softmax(old_rep / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        loss_incremental = self.weight * res
        return loss_incremental
      
      