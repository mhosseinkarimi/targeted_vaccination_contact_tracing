import numpy as np
import pandas as pd
import networkx as nx


# load data
data = pd.read_csv("data/school_dataset.csv")

num_nodes = np.max(np.max(data[["s", "d"]]))
nodes = np.arange(1, num_nodes+1)

# clusters
clusters = np.ones((num_nodes,))
# status
status = np.zeros((num_nodes,))
infected_idx = np.random.randint(1, num_nodes+1, (10,))
recovered_idx = np.random.randint(1, num_nodes+1, (10,))
status[infected_idx] = 1
status[recovered_idx] = 2

# vaccination prioirty
V = 25
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(data[["s", "d"]].to_numpy())

# finding priority
centrality = nx.centrality.degree_centrality(graph)
valid_nodes = nodes[status == 0]
valid_nodes_centrality = {n: centrality[n] for n in valid_nodes}
valid_nodes_centrality_rev = {val: key for key, val in valid_nodes_centrality.items()}

priorities = sorted(valid_nodes_centrality_rev, reverse=True)
priorities_dict = {valid_nodes_centrality_rev[val]: val for val in priorities[:V]}
priority_id = np.array(list(priorities_dict.keys()))-1

vaccination_priority = np.zeros((num_nodes,))
vaccination_priority[priority_id] = 1

users_info = pd.DataFrame({
    "id": nodes,
    "cluster": clusters,
    "status": status,
    "priority": vaccination_priority,
})

users_info.to_csv("data/users_info.csv", index=False)