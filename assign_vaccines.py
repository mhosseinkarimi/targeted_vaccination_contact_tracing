import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from utils.targeted_vaccination import *

# load clusters
with open("artifacts/clusters/vgae/gmm_clusters.pkl", "rb") as f:
    cluser_list = pickle.load(f)

# Finding the cluster populations
clusters, cluser_population = np.unique(cluser_list, return_counts=True)
n_clusters = len(clusters)

F_max, f_opt = optimize_obj_fcn(cluser_population, 100, i0_vec=[0.03, 0.01], tau=10, beta=0.5, gamma=0.1)
print(f"Vaccination Shares: First Cluster: {f_opt[0] * cluser_population[0]},\
      Second Cluster: {f_opt[1] * cluser_population[1]}")
