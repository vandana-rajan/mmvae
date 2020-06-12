# VAE for RECOLA audio arousal

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod, sqrt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from utils import Constants
from vis import plot_embeddings, plot_kls_df
from .vae import VAE

# Constants
dataSize = torch.Size([1, 88])  # audio arousal feature size
data_dim = int(prod(dataSize))
hidden_dim = 64

x_train = pd.read_csv('./audio_arousal/audio_arousal_train_x.txt',index_col=False,sep="\t",header=None).values[:,:-1]
x_dev = pd.read_csv('./audio_arousal/audio_arousal_dev_x.txt',index_col=False,sep="\t",header=None).values[:,:-1]

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for RECOLA audio arousal features. """
    def __init__(self, latent_dim):
        super(Enc, self).__init__()
        self.linear1 = nn.Linear(data_dim, hidden_dim)
        self.enc = nn.Linear(hidden_dim, latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
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

class AudioArousal(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """
    def __init__(self, params):
        super(AudioArousal, self).__init__(
            dist.log_normal.LogNormal,  # prior
            dist.log_normal.LogNormal,  # likelihood
            dist.log_normal.LogNormal,  # posterior
            Enc(params.latent_dim),
            Dec(params.latent_dim),
            params
        )
        grad = {'requires_grad': params.learn_prior}
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim), **grad)  # logvar
        ])
        self.modelName = 'audio_arousal'
        self.dataSize = dataSize
        self.llik_scaling = 1

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @staticmethod
    def getDataLoaders(batch_size):
        train = DataLoader(TensorDataset(torch.Tensor(x_train)),batch_size=batch_size,shuffle=True,num_workers=4)
        dev = DataLoader(TensorDataset(torch.Tensor(x_dev)),batch_size=batch_size,shuffle=False,num_workers=4)
        return train,dev

    def generate(self, runPath, epoch):
        N, K = 64, 9
        samples = super(AudioArousal, self).generate(N, K).cpu()
        return samples

    def reconstruct(self, data, runPath, epoch):
        recon = super(AudioArousal, self).reconstruct(data)
        return recon

    def analyse(self, data, runPath, epoch):
        zemb, zsl, kls_df = super(AudioArousal, self).analyse(data, K=10)
        labels = ['Prior', self.modelName.lower()]
        plot_embeddings(zemb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(runPath, epoch))
        plot_kls_df(kls_df, '{}/kl_distance_{:03d}.png'.format(runPath, epoch))
