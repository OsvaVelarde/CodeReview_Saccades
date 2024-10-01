import pickle as pkl
from pandas import read_csv
from pandas import DataFrame

import sys
sys.path.append( '/home/osvaldo/Documents/CCNY/Project_Saccades/src/utils/' )
from movie import time_sequence, rep_video

import matplotlib.pyplot as plt
import numpy as np

movies = {
    '01': 'The_Big_Sick',
    '02': 'The_Peanut_Butter',
    '03': 'Whiplash',
    '04': 'Room',
    '06': 'Me_Earl_and_Dying_Girl',
    '09': 'The_Tomorrow_Man',
    '11': 'Dom_Hemingway',
    '12': 'Life_After_Beth',
    '13': 'Woodshock',
    '14': 'The_Comedian'}

foldernames = {'event':'seq_event_cuts',
              'cont':'seq_cont_cuts',
              'fixed':'seq_fixed_len',
              'chapter':'seq_chapters'}

# --------------------------------------------------------------------
# --------------------------------------------------------------------

type_seq = 'chapter'
moviepath = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/'
csvpath  = moviepath + foldernames[type_seq]+ '/csv/'
datapath = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'

# --------------------------------------------------------------------
# -------------------- SELECTION OF SEQUENCES ------------------------
exp = 4
# thr_len_seq = 50
losspath = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/predictions/'

with open(losspath + 'exp_' + '{:02d}'.format(exp) + '_loss.pkl','rb') as f:
    data = pkl.load(f)

# --------------------------------------------------------------------
# --------------------------------------------------------------------

len_seqs = [len(seq) for _, seq in data.items()]

fig = plt.figure(figsize=(10,10), layout="constrained")
axs = fig.subplots(1, 1, sharex=True, sharey=True)

n_bins = 100
n, bins, patches = axs.hist(len_seqs, n_bins, density=True, histtype="step",
                               cumulative=-1, label="Cumulative histogram")

axs.set_xlabel("Len")
axs.set_ylabel("Cumulative Inv")

p = 90
sorted_len_seqs = sorted(len_seqs)
num_seqs = len(sorted_len_seqs)    
indice = int((100 - p) / 100 * num_seqs)  # Tomamos la parte entera
thr_len_seq = sorted_len_seqs[indice]
print('Threshold:', thr_len_seq)

# --------------------------------------------------------------------
# --------------------------------------------------------------------

data_filter = {idx[0]: seq[:thr_len_seq] for idx, seq in data.items() if len(seq)>=thr_len_seq}
loss_seqs = {idx: sum(seq)/thr_len_seq for idx, seq in data_filter.items()}

order_keys_loss = sorted(loss_seqs, key=loss_seqs.get)

df = DataFrame.from_dict(loss_seqs,orient='index')
df.to_csv('sortloss.csv')
print(df)

# for kk,vv in order_keys_loss.items():
#     print(kk, vv)

# min_value = loss_seqs[order_keys_loss[0]]
# max_value = loss_seqs[order_keys_loss[-1]]

# num_scenes   = 8

# best_or_worst = 'best'
# scenes = order_keys_loss[:num_scenes] if best_or_worst == 'best' else order_keys_loss[-num_scenes:]

# print('Min Loss:', min_value, 'Max Loss:', max_value)
# print('------------------------------------------------------------')

# # ----------------------------------------------------------------------
# # --------------------- PLOTTING VIDEOS --------------------------------

# for scene in scenes:
#     # ------------------------------------------------------------------
#     filename = scene[0][:-3]
    
#     if type_seq == 'chapter':
#         _, _, idx_part, _, idx_movie, _, _, idx_chap = filename.split('_')
#         movie_name = movies[idx_movie]
#         print('Participant:', idx_part, 'Movie:', idx_movie, movie_name, 'Chapter: ', idx_chap)
#     else:
#         _, _, idx_part, _, idx_movie, _, _, idx_chap, _, idx_scene = filename.split('_')
#         movie_name = movies[idx_movie]
#         print('Participant:', idx_part, 'Movie:', idx_movie, movie_name, 'Chapter: ', idx_chap, 'Seq:', idx_scene)
    
#     prefix_filename = 'metadata_participant_' + idx_part + '_stimuli_' + idx_movie

#     #print('Loss:',loss_seqs[scene])
#     # ------------------------------------------------------------------

#     stim_metadata = read_csv(datapath + 'data_python/' +  prefix_filename + '_stim.csv')
#     stim_initial = stim_metadata.iloc[int(idx_movie)-1]['stimStart']

#     avi_filename = datapath + 'stimuli/' + movie_name + '/' + movie_name + '-ch' + str(int(idx_chap)) + '_padded.avi'
    
#     if type_seq == 'chapter':
#         t_ini, t_end = time_sequence(datapath,type_seq, idx_part, idx_movie, idx_chap)
#     else:
#         t_ini, t_end = time_sequence(datapath,type_seq, idx_part, idx_movie, idx_chap, idx_scene)

#     # ------------------------------------------------------------------
#     rep_video(avi_filename, t_ini, t_end, stim_initial,csvpath + filename + '.csv')

#     print('------------------------------------------------------------')    