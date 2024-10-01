import random
import matplotlib.pyplot as plt

from dataset import SaccadesParraLab
from torch.utils.data import DataLoader, random_split
from torch import transpose

# ======================================================================
# ========================== DATASET ===================================

datapath = '/media/osvaldo/OMV5TB/Movie_Scenes/pts/'
max_len_seq = 10
n_seq_plot = 5

dataset = SaccadesParraLab(datapath)
subset_plot, _ = random_split(dataset, [n_seq_plot, len(dataset)-n_seq_plot])
loader = DataLoader(subset_plot, batch_size=1, shuffle=True, num_workers=1)

fig, axs = plt.subplots(n_seq_plot,max_len_seq, figsize=(20,5))

for batch_idx, batch in enumerate(loader):
    name_seq, seq_patches = batch

    for t in range(max_len_seq):
        patch = seq_patches[0,t,:,:,:]
        axs[batch_idx,t].imshow(patch.permute(1,2,0))

plt.show()