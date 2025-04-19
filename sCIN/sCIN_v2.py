import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.decomposition import PCA
from sCIN.utils import impute_cells
from itertools import cycle
from typing import *
import os


class Mod1Encoder(nn.Module):
    """ The first modality encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod1Encoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
    
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward computation in the network.

        Parameters
        ----------
        x : torch.tensor
            Mod1 data matrix
            
        Returns
        -------
        z_mod1 : torch.tensor
            Mod1 embeddings 
        """
        h = self.linear1(x)
        h1 = self.bn(h)
        h2 = self.relu(h1)
        z_mod1 = self.linear2(h2)

        return z_mod1
    

class Mod1Decoder(nn.Module):
    pass


class Mod2Decoder(nn.Module):
    pass
    

class Mod2Encoder(nn.Module):
    """ The second modality encoder with three layers"""
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Mod2Encoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward computation in the network.

        Parameters
        ----------
        x : torch.tensor
            Mod2 count tensor

        Returns
        -------
        z_mod2 : torch.tensor
            Mod2 embeddings 
        """
        h = self.linear1(x)
        h1 = self.bn(h)  
        h2 = self.relu(h1)
        z_mod2 = self.linear2(h2)

        return z_mod2
    
    
class sCINV1(nn.Module):
    """Main sCIN class."""
    def __init__(self,
                 num_mod1_samples: int,
                 num_mod1_features: int,
                 num_mod2_samples: int,
                 num_mod2_features: int,
                 num_cell_types: int,
                 mod1_hidden_dim: int = 256,
                 mod2_hidden_dim: int = 256,
                 latent_dim: int = 128,
                 t: int = 0.07,
                 sCIN_configs: Dict[str, int] = None):
        super(sCINV1, self).__init__()

        self.num_mod1_samples = num_mod1_samples
        self.num_mod2_samples = num_mod2_samples
        self._num_samples = num_mod1_features + num_mod2_features
        self.latent_dim = latent_dim 


        # Initialize encoders and params
        self.mod1_encoder = Mod1Encoder(num_mod1_features,
                                        mod1_hidden_dim,
                                        self.latent_dim)
        
        self.mod2_encoder = Mod2Encoder(num_mod2_features,
                                        mod2_hidden_dim,
                                        self.latent_dim)

        self.cell_types_embs = nn.Embedding(num_cell_types, 
                                            self.latent_dim)
        
        self.v = nn.Linear(self._num_samples, self.latent_dim)
        self.t = nn.Parameter(torch.tensor(t))
        self.W_mod1 = nn.Parameter(torch.randn((self.latent_dim, 
                                                self.latent_dim)))
        self.W_mod2 = nn.Parameter(torch.randn((self.latent_dim, 
                                                self.latent_dim)))
    

    def _compute_attn_weights(q: torch.Tensor, 
                              k: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights from the query and the key matrices.
        This is classic scaled dot product from Vaswani et al. (2017):

        softmax(\frac^{QK^T}_{\sqrt{d_k}})
        """
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k.T) / d_k**0.5
        weights = F.softmax(scores, dim=-1)

        return weights
        

    def forward(self, x_mod1, x_mod2):
        """
        Forward computation in the network.
        """
        z_mod1 = self.mod1_encoder(x_mod1)  # N * d
        z_mod2 = self.mod2_encoder(x_mod2)  # M * d

        z_joint = torch.cat([z_mod1, z_mod2], dim=0)  # (M + N) * d

        q = torch.matmul(z_joint, self.cell_types_embs.T)  # (M + N) * k
        k = torch.matmul(z_joint, self.cell_types_embs.T)  # (M + N) * k

        attn_weights = self._compute_attn_weights(q, k)  # (M + N) * (M + N)
        z_joint_ = self.v(attn_weights)  # (M + N) * d
        z_mod1_ = z_joint_[self._num_samples:]
        z_mod2_ = z_joint_[:self._num_samples]

        mod1_embs = F.normalize(torch.matmul(z_mod1_, self.W_mod1), p=2, dim=1)
        mod2_embs = F.normalize(torch.matmul(z_mod2_, self.W_mod2), p=2, dim=1)
        logits = torch.matmul(mod1_embs, mod2_embs.t()) * torch.exp(self.t)

        return logits, mod1_embs, mod2_embs
    

def train_sCIN_v2():
    pass



def get_embs_sCIN_v2():
    pass