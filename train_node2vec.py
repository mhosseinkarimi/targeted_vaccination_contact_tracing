import pickle
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.nn import Node2Vec

from prepare_data import create_bbc_dataset

warnings.filterwarnings("ignore")


# parsing parameters from command line
parser = ArgumentParser()

parser.add_argument('-p', default=1, type=float, 
                    help="Likelihood of immediately revisiting a node in the walk.")
parser.add_argument('-q', default=1, type=float, 
                    help="Control parameter to interpolate between breadth-first strategy and depth-first strategy")
parser.add_argument("--embedding_dim", default=128, type=int,
                    help="The size of each embedding vector.")
parser.add_argument("--walk_length", default=20, type=int,
                    help="Number of nodes visited in every walk.")
parser.add_argument("--context_size", default=10, type=int,
                    help=""" The actual context size which is considered for positive samples. 
                    This parameter increases the effective sampling rate by reusing samples across different source nodes.""")
parser.add_argument("--lr", default=0.01, type=float,
                    help="Learning rate")
parser.add_argument("--epoch", default=50, type=int,
                    help="Number of epochs.")
parser.add_argument("--save_model", action="store_true",
                    help="The option to save the model.")
parser.add_argument("--model_save_dir", default="artifacts/models/node2vec", type=str,
                    help="The path to saved model directory.")
parser.add_argument("--plot_points", action="store_true",
                    help="The option of plotting the embedded nodes.")

args = parser.parse_args()

# Load dataset
data, grpah = create_bbc_dataset('data/bbc.csv')

# initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    data.edge_index,
    embedding_dim=args.embedding_dim,
    walk_length=args.walk_length,
    context_size=args.context_size,
    walks_per_node=10,
    num_negative_samples=1,
    p=args.p,
    q=args.q,
    sparse=True,
    ).to(device)

num_workers = 0 if sys.platform.startswith('win') else 4
loader = model.loader(batch_size=128, shuffle=True,
                    num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

# training method
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# @torch.no_grad()
# def test():
#     model.eval()
#     z = model()
#     acc = model.test(z[data.train_mask], data.y[data.train_mask],
#                      z[data.test_mask], data.y[data.test_mask],
#                      max_iter=150)
#     return acc

# training process
loss_list = []
for epoch in range(1, args.epoch + 1):
    loss = train()
    loss_list.append(loss)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


# performance plots
font = {'family':'serif','color':'darkred','size':15}
font_title = {'family':'serif','color':'darkred','size':25}

fig_loss = plt.figure(figsize=(15, 8))
plt.plot(np.arange(1, args.epoch+1), loss_list, "g--o", label="Loss")
plt.title("Loss for Node2Vec", fontdict=font_title)
plt.xlabel("Epoch", fontdict=font)
plt.ylabel("Loss Value", fontdict=font)
plt.grid("minor")
plt.savefig("/home/hossein/Projects/Covid Contact Tracing/artifacts/figures/node2vec/loss.png")

# plotting the embedded nodes
@torch.no_grad()
def plot_points():
    model.eval()
    z = model(torch.arange(data.num_nodes, device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())

    plt.figure(figsize=(8, 8))
    plt.scatter(z[:, 0], z[:, 1])
    plt.axis('off')
    plt.savefig("artifacts/figures/node2vec/node2vec-embedding.png")

if args.plot_points:
    plot_points()

if args.save_model:
    # saving model setup
    train_datetime = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    with open(f"{args.model_save_dir}/setup_node2vec_{train_datetime}.pkl", "wb") as f:
        pickle.dump(args, f)
    # saving model state
    torch.save(model.state_dict(), 
               f"{args.model_save_dir}/model_node2vec_{train_datetime}.pt")
    print(f"Model was saved in: " + f"{args.model_save_dir}/model_node2vec_{train_datetime}.pt")