# VAE for RECOLA audio arousal

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import DataLoader

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
dataSize = torch.Size([1, 88]) # audio arousal feature size
data_dim = int(prod(dataSize))
hidden_dim = 64

# Classes
class Enc(nn.Module):
  """ Generate latent parameters for RECOLA audio arousal features. """
  def __init__(self, latent_dim):
    super(Enc, self).__init__()
    self.linear1 = nn.Linear(data_dim, hidden_dim)
    self.enc = nn.Linear(hidden_dim, latent_dim)
    self.fc21 = nn.Linear(latent_dim,latent_dim)
    self.fc22 = nn.Linear(latent_dim,latent_dim)
  def forward(self,x):
    e = F.tanh(self.enc(F.tanh(self.linear1(x))))  # x.shape=(batch_size,88)
    lv = self.fc22(e)
    return self.fc21(e), F.softmax(lv, dim=-1) * lv.size(-1) + Constants.eta
 
class Dec(nn.Module):
    """ Generate RECOLA audio arousal feature given a sample from the latent space. """
    def __init__(self, latent_dim):
      super(Dec, self).__init__()
      self.dec = nn.Linear(latent_dim, hidden_dim)
      self.fc3 = nn.Linear(hidden_dim, data_dim)
    def forward(self, z):
      p = self.fc3(F.tanh(self.dec(z)))
      return p, torch.tensor(0.75).to(z.device)  # mean, length scale
 
      

    

