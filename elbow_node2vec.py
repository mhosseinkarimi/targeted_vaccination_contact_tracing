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
from yellowbrick.cluster import KElbowVisualizer

from model import get_gae, get_vgae
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

# Clustering with K-Means
visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='distortion')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_node2vec_distortion.png")
plt.clf()

visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='silhouette')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_node2vec_silhouette.png")
plt.clf()

visualizer = KElbowVisualizer(KMeans(), k=(3,12), metric='calinski_harabasz')
visualizer.fit(latent_numpy)
visualizer.finalize()
plt.savefig(f"artifacts/figures/elbow/kmeans_node2vec_calinski_harabasz.png")
plt.clf()