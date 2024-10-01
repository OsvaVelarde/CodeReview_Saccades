import torch
import os
import matplotlib.pyplot as plt
import pickle
import random

predictions_folder = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/predictions/'
title  = 'exp_03'

# -------------------------------------------------------------------

with open(predictions_folder + title + '.pkl', 'rb') as f:
    data = pickle.load(f)

claves = list(data.keys())
claves_aleatorias = random.sample(claves, 2)
elementos_aleatorios = {clave: data[clave] for clave in claves_aleatorias}

print(elementos_aleatorios)
exit()
list_names = []
list_values = []

for kk, vv in data.items():
    list_names.append(kk[0])
    list_values.extend(vv)

predictions = torch.cat(list_values, dim=0)

# -------------------------------------------------------------------

mean = torch.mean(predictions,dim=0)
std = torch.std(predictions,dim=0)
print(predictions)

exit()
#plt.plot(mean.cpu())
plt.fill_between(torch.arange(1, 129),(mean-std).cpu(),(mean+std).cpu())
plt.show()
#print(mean.shape)
# lens_batch = []
# means_batch = []
# vars_batch = []

# for ff in filename_batchs:
# 	batch_preds = torch.load(predictions_folder + ff)
# 	lens_batch.append(batch_preds.shape[0])
# 	means_batch.append(batch_preds.mean(dim=0))
# 	vars_batch.append(batch_preds.var(dim=0, keepdim=True).squeeze())

# means_batch = torch.stack(means_batch)
# lens_batch = torch.tensor(lens_batch, dtype=means_batch.dtype, device=means_batch.device)
# vars_batch = torch.stack(vars_batch)
# len_total = torch.sum(lens_batch)
# 

# weight_means = means_batch * lens_batch.view(-1, 1)  
# mean = torch.sum(weight_means, dim=0) / len_total

# # -------------------------------------------------------------------

# weight_vars = vars_batch * lens_batch.view(-1, 1)
# first_var = torch.sum(weight_vars, dim=0) / len_total

# weight_means_sqr = torch.pow(means_batch, 2) * lens_batch.view(-1, 1)
# second_var = torch.sum(weight_means_sqr, dim=0) / len_total

# var = first_var + second_var - torch.pow(mean,2)

# # -------------------------------------------------------------------

# plt.plot(var.cpu())
# plt.show()