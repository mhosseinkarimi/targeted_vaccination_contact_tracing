import pickle
import warnings
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")

from prepare_data import create_bbc_dataset

# parsing parameters from command line
parser = ArgumentParser()

parser.add_argument('--model_name', type=str, choices={"node2vec", "gae", "vgae"},
                    help="name of model")
args = parser.parse_args()

# load data
data, graph = create_bbc_dataset("data/bbc.csv")

# load clusters
with open(f"artifacts/clusters/{args.model_name}/kmeans_clusters.pkl", 'rb') as f:
    kmeans_clusters = pickle.load(f)
    
with open(f"artifacts/clusters/{args.model_name}/dbscan_clusters.pkl", 'rb') as f:
    dbscan_clusters = pickle.load(f) 

with open(f"artifacts/clusters/{args.model_name}/gmm_clusters.pkl", 'rb') as f:
    gmm_clusters = pickle.load(f)

# plot graphs
fig = plt.figure(figsize=(35, 40))
ax = fig.add_axes([0, 0, 1, 1])
nx.draw_networkx(graph, ax=ax, node_color=kmeans_clusters, with_labels=False, edge_color="#D3D3D3")
plt.savefig(f'artifacts/figures/{args.model_name}/kmean_nodes_clusters.png')

fig = plt.figure(figsize=(35, 40))
ax = fig.add_axes([0, 0, 1, 1])
nx.draw_networkx(graph, ax=ax, node_color=gmm_clusters, with_labels=False, edge_color="#D3D3D3")
plt.savefig(f'artifacts/figures/{args.model_name}/gmm_nodes_clusters.png')

fig = plt.figure(figsize=(35, 40))
ax = fig.add_axes([0, 0, 1, 1])
nx.draw_networkx(graph, ax=ax, node_color=dbscan_clusters, with_labels=False, edge_color="#D3D3D3")
plt.savefig(f'artifacts/figures/{args.model_name}/dbscan_nodes_clusters.png')