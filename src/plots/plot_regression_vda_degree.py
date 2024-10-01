# vda = K * px + int

import json
import matplotlib.pyplot as plt
import numpy as np

filename = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/coeff_px_vda.json'
with open(filename, "r") as jsonfile:
    coeffs = json.load(jsonfile)

factors = [720./1920.,404./1080.]

data = []
for kk, vv in coeffs.items():
	for cc, vv_cc in vv.items():
		data.append(vv_cc)

pixels = np.linspace(0,2000,2000)
fig, ax = plt.subplots(1,4,figsize=(20,5))

for idx, row in enumerate(data):
	vda_x = row[0] * pixels
	vda_y = row[2] * pixels

	ax[0].plot(pixels,vda_x)
	ax[1].plot(pixels,vda_y)

dvda = 5 
dpx = np.array([[dvda/row[0], dvda/row[2]] for row in data])
means = np.mean(dpx,axis=0)

for ii in range(2):
	ax[2].plot(dpx[:,ii],label='Monitor mean = ' + str(int(means[ii])))
	ax[2].legend()

	ax[3].plot(dpx[:,ii]*factors[ii],label='Video mean = ' + str(int(means[ii]*factors[ii])))
	ax[3].legend()

plt.show()
#data = np.array(data)
