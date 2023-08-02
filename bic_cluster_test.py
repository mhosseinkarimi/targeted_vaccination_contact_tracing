import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.mixture import GaussianMixture

from model import get_gae, get_vgae
from prepare_data import create_bbc_dataset

warnings.filterwarnings("ignore")


# parsing parameters from command line
parser = ArgumentParser()

parser.add_argument('--AE_type', default="gae", type=str, choices={"gae", "vgae"},
                    help="Type of Graph autoencoder for embedding.")
args = parser.parse_args()


# data loading
dataset, graph = create_bbc_dataset('data/bbc.csv')

# transformation of data
transform = T.NormalizeFeatures(attrs=["x", "edge_attr"])
dataset = transform(dataset)

# parameters
use_edge_weight = False
in_channels = dataset.num_features
latent_channels = 256

# clustering
if use_edge_weight:
    edge_weight = dataset.edge_attr
    print(edge_weight)
else:
    edge_weight = None

# load model
if args.AE_type == "gae":
    model = get_gae(in_channels, latent_channels)
    model.load_state_dict(torch.load('artifacts/models/gae/gcn_gae.pt'))
    model.eval()
else:
    model = get_vgae(in_channels, latent_channels)
    model.load_state_dict(torch.load('artifacts/models/vgae/gcn_vgae.pt'))
    model.eval()

# encoding graph into latent space
latent_space = model.encode(dataset.x, dataset.edge_index, edge_weight)
latent_numpy = latent_space.detach().numpy()

# Testing BIC
n_components = range(1, 15)
covariance_type = ['spherical', 'tied', 'diag', 'full']
score=[]
for cov in covariance_type:
    for n_comp in n_components:
        gmm=GaussianMixture(n_components=n_comp,covariance_type=cov, reg_covar=0.001)
        gmm.fit(latent_numpy)
        score.append([cov,n_comp,gmm.bic(latent_numpy)])
        
        
best_score_arg = np.argmin(np.array(score)[:, 2])
print(f"Cov: {score[best_score_arg][0]}, n_comp: {score[best_score_arg][1]}, Score: {score[best_score_arg][2]}")