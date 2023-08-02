from torch.nn import ReLU, Dropout
from torch_geometric.nn import GCNConv, Sequential
import torch
from torch_geometric.nn.models import GAE, VGAE


class GCNConvEncoder(torch.nn.Module):
    """An encoder module based on GCN convolution layers introduced at:
    Semi-Supervised Classification with Graph Convolutional Networks
    By: Thomas N. Kipf, Max Welling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = Sequential("x, edge_index, edge_weight", [
            (GCNConv(in_channels, 2*out_channels, cached=True), "x, edge_index, edge_weight -> x"),
            # (Dropout(), "x -> x"),
            (ReLU(inplace=True), "x -> x"),
            (GCNConv(2*out_channels, out_channels, cached=True), "x, edge_index, edge_weight -> x")
        ])

    def forward(self, x, edge_index, edge_weight=None):
        return self.encoder(x, edge_index, edge_weight)


class VariationalGCNConvEncoder(torch.nn.Module):
    """An encoder module based on GCN convolution layers introduced at:
    Semi-Supervised Classification with Graph Convolutional Networks
    By: Thomas N. Kipf, Max Welling
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels, cached=True)
        self.conv_mu = GCNConv(2*out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2*out_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(x, edge_index, edge_weight)


def get_gae(
    in_channels, out_channels, 
    encoder=GCNConvEncoder,
    decoder=None
            ):
    """Constructs a GAE."""
    return GAE(encoder(in_channels, out_channels), decoder)


def get_vgae(
    in_channels, out_channels, 
    encoder=VariationalGCNConvEncoder,
    decoder=None
    ):
    """Constructs a Variational Graph Autoencoder."""
    return VGAE(encoder(in_channels, out_channels), decoder)