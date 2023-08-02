import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

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
latent_channels = 200

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


# Clustering with K-Means
visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='distortion')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_{args.AE_type}_distortion.png")
plt.clf()

visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='silhouette')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_{args.AE_type}_silhouette.png")
plt.clf()

visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='calinski_harabasz')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_{args.AE_type}_calinski_harabasz.png")
plt.clf()