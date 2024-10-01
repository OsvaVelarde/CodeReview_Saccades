import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import stack, cat, argsort
from torch.linalg import eigh

PATH = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/'
criteria = ['movie','chapter']

# --------------------------------------------------------------------
exp = 4
datafolder = PATH + 'predictions/'
plotfolder = PATH + 'plots/exp_' + '{:02d}'.format(exp) + '/'

with open(datafolder + 'exp_' + '{:02d}'.format(exp) + '.pkl','rb') as f:
    predictions = pkl.load(f)

print('Data Loaded')
# --------------------------------------------------------------------

idx_seq = []
preds_seq = []

for kk, vv in predictions.items():
    info_seq = kk[0][:-3].split('_')
    idx_seq.append([int(info_seq[2]),int(info_seq[4]),int(info_seq[7])]) # part - movie - chap

    preds = stack(vv).squeeze(1).transpose(0, 1)
    preds_seq.append(preds)

# --------------------------------------------------------------------

df_info = pd.DataFrame(idx_seq,columns=['participant','movie','chapter'])
groups = df_info.groupby(criteria)

# --------------------------------------------------------------------

data = cat(preds_seq, dim=1)
num_samples = data.shape[1]
data_mean = data.mean(dim=1, keepdim=True)
data_cent = data - data_mean

cov_matrix = data_cent @ data_cent.T / (num_samples - 1)
eigvals, eigvecs = eigh(cov_matrix)
sorted_indices = argsort(eigvals, descending=True)
transformation = eigvecs[:, sorted_indices[:2]]

# ------------------------------------------------------------------------

for group_idx, df_gg in groups:
    
    fig, axs = plt.subplots(1,1,figsize=(10,10))
    title = 'Movie_' + str(group_idx[0]) + '_Chapter_' + str(group_idx[1])
    
    preds_idxs = list(df_gg.index.values)
    reduced_preds_seq = []

    for ii in preds_idxs:
        traj = preds_seq[ii] - preds_seq[ii].mean(dim=1, keepdim=True)
        traj = transformation.T @ traj
        traj = traj.cpu().numpy()
        axs.plot(traj[0,:], traj[1,:], '-')

    fig.savefig(plotfolder + 'Trajectory_' + title + '.svg')
    plt.close(fig)        
