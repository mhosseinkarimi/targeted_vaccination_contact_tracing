import pickle
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
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

# Clustering with DBSCAN
dbscan_clusters = DBSCAN(eps=0.1, min_samples=10).fit_predict(latent_numpy)
with open(f"artifacts/clusters/{args.AE_type}/dbscan_clusters.pkl", 'wb') as f:
    pickle.dump(dbscan_clusters, f)
# Clustering with K-Means
# kmeans_clusters = GapStatClustering(base_clusterer=KMeans(), max_k=100).fit_predict(latent_numpy)
kmeans_clusters = KMeans(n_clusters=3).fit_predict(latent_numpy)
with open(f"artifacts/clusters/{args.AE_type}/kmeans_clusters.pkl", 'wb') as f:
    pickle.dump(kmeans_clusters, f)
# Clustering with GMM
gmm_clusters = GaussianMixture(n_components=3, covariance_type="spherical").fit_predict(latent_numpy)
with open(f"artifacts/clusters/{args.AE_type}/gmm_clusters.pkl", 'wb') as f:
    pickle.dump(gmm_clusters, f)
# TSNE
latent_tsne = TSNE(n_components=2).fit_transform(latent_space.detach().numpy())

font = {'family':'serif','color':'darkred','size':15}
font_title = {'family':'serif','color':'darkred','size':25}
colors = ["blue", "yellow", "red", "green", "orange", "gray", "purple"]
# visualizing clusters: DBSCAN
n_clusters = np.max(dbscan_clusters) + 2
cmap = plt.cm.get_cmap('hsv', n_clusters)
for k in range(-1, n_clusters+1):
    x, y = latent_tsne[np.where(dbscan_clusters == k)[0]].T
    plt.scatter(x, y, c=colors[k])
plt.title(f"DBSCAN Clustering with {args.AE_type.upper()}", fontdict=font_title)
plt.savefig(f"artifacts/figures/{args.AE_type}/dbscan_clusters.png")

# visualizing clusters: K-Means
n_clusters = np.max(kmeans_clusters) + 1
print(n_clusters)
# n_clusters = kmeans_clusters.n_clusters_
cmap = plt.cm.get_cmap('hsv', n_clusters)
for k in range(n_clusters):
    x, y = latent_tsne[np.where(kmeans_clusters == k)[0]].T
    plt.scatter(x, y, c=colors[k])
plt.title(f"K-Means Clustering with {args.AE_type.upper()}", fontdict=font_title)
plt.savefig(f"artifacts/figures/{args.AE_type}/kmeans_clusters.png")

# visualizing clusters: GMM
n_clusters = np.max(gmm_clusters) + 1
cmap = plt.cm.get_cmap('hsv', n_clusters)
for k in range(n_clusters):
    x, y = latent_tsne[np.where(gmm_clusters == k)[0]].T
    plt.scatter(x, y, c=colors[k])
plt.title(f"GMM Clustering with {args.AE_type.upper()}", fontdict=font_title)
plt.savefig(f"artifacts/figures/{args.AE_type}/gmm_clusters.png")
