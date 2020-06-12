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
  

