import torch
import matplotlib.pyplot as plt


filename = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/checkpoints/ckpt_exp_05.pth'

checkpoint = torch.load(filename)
loss = checkpoint['loss']
epoch = checkpoint['epoch']
weights = checkpoint['model']

fig, axs = plt.subplots(1,2,figsize=(10,5))

# Projector -------------------------------
print('projector')
head_layers = [weights['module.encoder.head.' + str(3*i) + '.weight'] for i in range(3)]
#time_const = weights['module.encoder.rnn.temporal_parameter']

for hh in head_layers:
	print('shape: ', hh.shape)
	min_v = hh.min().item()
	max_v = hh.max().item()
	print('min: ', min_v)
	print('max: ', max_v)
	axs[0].hist(hh.reshape(-1).cpu().numpy(), bins=1000, range=(min_v, max_v))

# Predictor -------------------------------
print('predictor')
predictor_layers = [weights['module.predictor.' + str(3*i) + '.weight'] for i in range(2)]

for hh in predictor_layers:
	print('shape: ', hh.shape)
	min_v = hh.min().item()
	max_v = hh.max().item()
	print('min: ', min_v)
	print('max: ', max_v)
	axs[1].hist(hh.reshape(-1).cpu().numpy(), bins=1000, range=(min_v, max_v))

for aa in axs:
	aa.set_xlim(-0.1,0.1)

plt.show()