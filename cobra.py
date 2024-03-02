import torch
import numpy as np
from torch import nn, einsum
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from math import ceil
from mamba import Mamba, MambaConfig



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)






class COBRAGatingUnit(nn.Module):
    def __init__(self,d_model,d_ffn,dropout):
        super().__init__()
        
        self.config = MambaConfig(d_model=d_model, n_layers=1)
    
        self.COB_1 = Mamba(self.config)

        self.COB_2 = Mamba(self.config)
	
       

    def forward(self, x):
        u, v = x, x 
        u = self.COB_1(u)  
        v = self.COB_2(v)
        out = u * v
        return out


class COBRABlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.cobgu = COBRAGatingUnit(d_model,d_ffn,dropout)
        self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.cobgu(x)   
        x = x + residual      
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        out = x + residual
        return out









class COBRA(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            
            *[COBRABlock(d_model,d_ffn,dropout) for _ in range(num_layers)],
            
            
        )

    def forward(self, x):
        
        x = self.model(x)
        
        return x







