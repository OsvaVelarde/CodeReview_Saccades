import torch
import matplotlib.pyplot as plt


filename = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/checkpoints/ckpt_exp_05.pth'

checkpoint = torch.load(filename)
loss = checkpoint['loss']
epoch = checkpoint['epoch']
weights = checkpoint['model']

fig, axs = plt.subplots(1,2,figsize=(10,5))

# Predictor -------------------------------
predictor = weights['module.predictor.3.weight']

print('shape: ', predictor.shape)
min_v = predictor.min().item()
max_v = predictor.max().item()
print('min: ', min_v)
print('max: ', max_v)

axs[0].hist(predictor.reshape(-1).cpu().numpy(), bins=1000, range=(min_v, max_v))
im = axs[1].imshow(predictor.cpu().numpy(), cmap='viridis', aspect='auto', vmin=-0.01, vmax=0.01)
cbar = fig.colorbar(im, orientation='horizontal')

#axs[1].set_colorbar()

axs[0].set_xlim(-0.1,0.1)
# -----------------------------------------

print(predictor[:,0])
norms = torch.norm(predictor, dim=0)
print(norms)

plt.show()

