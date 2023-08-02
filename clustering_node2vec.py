import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch_geometric.nn import Node2Vec

from prepare_data import create_bbc_dataset

warnings.filterwarnings("ignore")

# data loading
dataset, graph = create_bbc_dataset('data/bbc.csv')

# # transformation of data
# transform = T.NormalizeFeatures(attrs=["x", "edge_attr"])
# dataset = transform(dataset)

# # parameters
# use_edge_weight = False
# in_channels = dataset.num_features
# latent_channels = 10

# # clustering
# if use_edge_weight:
#     edge_weight = dataset.edge_attr
#     print(edge_weight)
# else:
#     edge_weight = None

# load model
# loading model configuration
with open("artifacts/models/node2vec/setup_node2vec_07-27-2023-02-59-33.pkl", "rb") as f:
    args = pickle.load(f)

# initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    dataset.edge_index,
    embedding_dim=args.embedding_dim,
    walk_length=args.walk_length,
    context_size=args.context_size,
    walks_per_node=10,
    num_negative_samples=1,
    p=args.p,
    q=args.q,
    sparse=True,
    ).to(device)

model.load_state_dict(torch.load('artifacts/models/node2vec/model_node2vec_07-27-2023-02-59-33.pt'))
model.eval()

# encoding graph into latent space
latent_space = model(torch.arange(dataset.num_nodes, device=device))
latent_space = latent_space.cpu()
latent_numpy = latent_space.detach().numpy()

# Clustering with DBSCAN
dbscan_clusters = DBSCAN(eps=0.1, min_samples=10).fit_predict(latent_numpy)
with open("artifacts/clusters/node2vec/dbscan_clusters.pkl", 'wb') as f:
    pickle.dump(dbscan_clusters, f)
# Clustering with K-Means
kmeans_clusters = KMeans(n_clusters=6).fit_predict(latent_numpy)
with open("artifacts/clusters/node2vec/kmeans_clusters.pkl", 'wb') as f:
    pickle.dump(kmeans_clusters, f)
# Clustering with GMM
gmm_clusters = GaussianMixture(n_components=2, covariance_type="spherical").fit_predict(latent_numpy)
with open("artifacts/clusters/node2vec/gmm_clusters.pkl", 'wb') as f:
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
plt.title(f"DBSCAN Clustering with Node2Vec", fontdict=font_title)
plt.savefig("artifacts/figures/node2vec/dbscan_clusters.png")


# visualizing clusters: K-Means
n_clusters = np.max(kmeans_clusters) + 1
cmap = plt.cm.get_cmap('hsv', n_clusters)
for k in range(n_clusters):
    x, y = latent_tsne[np.where(kmeans_clusters == k)[0]].T
    plt.scatter(x, y, c=colors[k])
plt.title(f"K-Means Clustering with Node2Vec", fontdict=font_title)
plt.savefig("artifacts/figures/node2vec/kmeans_clusters.png")

# visualizing clusters: GMM
n_clusters = np.max(gmm_clusters) + 1
cmap = plt.cm.get_cmap('hsv', n_clusters)
for k in range(n_clusters):
    x, y = latent_tsne[np.where(gmm_clusters == k)[0]].T
    plt.scatter(x, y, c=colors[k])
plt.title(f"GMM Clustering with Node2Vec", fontdict=font_title)
plt.savefig("artifacts/figures/node2vec/gmm_clusters.png")