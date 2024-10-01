import os
import pandas as pd

path_data = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'

list_filenames = [ff.split('.')[0] for ff in os.listdir(path_data + 'metadata')]

dist_num_sacc = []
for ff in list_filenames:
	try:
		saccades = pd.read_csv(path_data + 'data_python/' + ff + '_sacc.csv')
		dist_num_sacc.append(len(saccades))
	except:
		continue

print(dist_num_sacc)
print(min(dist_num_sacc), max(dist_num_sacc), sum(dist_num_sacc))