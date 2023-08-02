import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

from model import get_gae
from prepare_data import create_bbc_dataset

warnings.filterwarnings("ignore")


# data loading
dataset, graph = create_bbc_dataset('data/bbc.csv')

# transformation of data
transform = T.NormalizeFeatures(attrs=["x", "edge_attr"])
dataset = transform(dataset)

# train test split
dataset = train_test_split_edges(dataset)

# parameters
latent_channels = 256
in_channels = dataset.num_features
epochs = 50
use_edge_weight = False

# model
model = get_gae(in_channels, latent_channels)

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = dataset.x.to(device)
train_pos_edge_index = dataset.train_pos_edge_index.to(device)
if use_edge_weight:
    train_pos_edge_attr = dataset.train_pos_edge_attr.to(device)
else:
    train_pos_edge_attr = None

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# defining train and test process
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index, train_pos_edge_attr)
    return model.test(z, pos_edge_index, neg_edge_index)

highest_ap, highest_auc = (0, 0), (0, 0)
ap_list = []
auc_list = []
loss_list = []

for epoch in range(1, epochs + 1):
    # backpropagation
    loss = train()
    loss_list.append(loss)
    # validation
    auc, ap = test(dataset.test_pos_edge_index, dataset.test_neg_edge_index)
    auc_list.append(auc)
    ap_list.append(ap)
    
    # logging the highest accuracy amd AUC
    if highest_ap[0] < ap:
        highest_ap = (ap, epoch)
        
    if highest_auc[0] < auc:
        highest_auc = (auc, epoch)
    
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

print(f"Highest accuracy: {highest_ap[0]}, Epoch: {highest_ap[1]}")
print(f"Highest AUC: {highest_auc[0]}, Epoch: {highest_auc[1]}")

# performance plots
font = {'family':'serif','color':'darkred','size':15}
font_title = {'family':'serif','color':'darkred','size':25}

fig_loss = plt.figure(figsize=(15, 8))
plt.plot(np.arange(1, epochs+1), loss_list, "g--o", label="Loss")
plt.title("Reconstruction Loss for GAE", fontdict=font_title)
plt.xlabel("Epoch", fontdict=font)
plt.ylabel("Loss Value", fontdict=font)
plt.grid("minor")
plt.savefig("/home/hossein/Projects/Covid Contact Tracing/artifacts/figures/gae/loss.png")

fig_auc = plt.figure(figsize=(15, 8))
plt.plot(np.arange(1, epochs+1), auc_list, "b--o", label="AUC")
plt.title("Area Under Curve for GAE", fontdict=font_title)
plt.xlabel("Epoch", fontdict=font)
plt.ylabel("Score", fontdict=font)
plt.grid("minor")
plt.savefig("/home/hossein/Projects/Covid Contact Tracing/artifacts/figures/gae/auc_score.png")

fig_ap = plt.figure(figsize=(15, 8))
plt.plot(np.arange(1, epochs+1), ap_list, "r--o", label="Accuracy")
plt.title("Accuracy of Reconstruction for GAE", fontdict=font_title)
plt.xlabel("Epoch", fontdict=font)
plt.ylabel("Score", fontdict=font)
plt.grid("minor")
plt.savefig("/home/hossein/Projects/Covid Contact Tracing/artifacts/figures/gae/ap_score.png")


# save model
torch.save(model.state_dict(), 'artifacts/models/gae/gcn_gae.pt')
print("Model was saved in: artifacts/models/gae/gcn_gae.pt")