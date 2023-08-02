import pickle
import warnings

import numpy as np
import torch
import torch_geometric.transforms as T
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