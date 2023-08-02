import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import tensor
from torch_geometric.data import Data


def format_csv_file(old_csv_path, new_csv_path="./data/bbc.csv"):
    """Reformat the BBC_dataset CSV file to a new one with new columns' names."""
    data_df = pd.read_csv(old_csv_path)
    data_df.rename(columns={"ID1": "source", "ID2": "target"}, inplace=True)
    data_df[["source", "target"]] -= 1
    data_df.to_csv(new_csv_path)


def create_bbc_dataset(data_path):
    """Create a torch_geometric Data object from the dataset."""
    # read data
    data_df = pd.read_csv(data_path)

    # construct the graph
    N = np.max(np.max(data_df[["source", "target"]]))
    G = nx.MultiGraph()
    G.add_nodes_from(np.arange(0, N+1))
    G.add_edges_from(np.array(data_df[["source", "target"]]))

    # using the degree of each node as its feature
    degree = G.degree()
    x = tensor([d for (n, d) in degree], dtype=torch.float32).reshape(-1, 1)

    return Data(
        x=x,
        edge_index=tensor(data_df[["source", "target"]].to_numpy()).T,
        edge_attr=tensor(data_df["distance"].to_numpy(), dtype=torch.float32)), G


# To reformat the dataset
if __name__ == "__main__":
    # format_csv_file('data/BBC_dataset.csv')
    dataset, graph = create_bbc_dataset("data/bbc.csv")
    print(dataset)