import pickle as pkl
from pandas import read_csv

import sys
sys.path.append( '/home/osvaldo/Documents/CCNY/Project_Saccades/src/utils/' )
from movie import time_sequence, rep_video

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
              'fixed':'seq_fixed_len'}

# --------------------------------------------------------------------
# --------------------------------------------------------------------

type_seq = 'event'
moviepath = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/'
csvpath  = moviepath + foldernames[type_seq]+ '/csv/'
datapath = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'

# --------------------------------------------------------------------
# -------------------- SELECTION OF SEQUENCES ------------------------
epoch = 49
exp = 5
thr_len_seq = 50
losspath = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/loss_recurrence/'
best_or_worst = 'best'

# --------------------------------------------------------------------
# --------------------------------------------------------------------

with open(losspath + 'exp_' + '{:02d}'.format(exp) + '/epoch_'+ str(epoch) + '.pkl','rb') as f:
    data = pkl.load(f)

len_seqs = {idx: len(seq) for idx, seq in data.items() if len(seq)>=thr_len_seq}
data_filter = {idx: seq[:thr_len_seq] for idx, seq in data.items() if idx in len_seqs}
loss_seqs = {idx: sum(seq)/thr_len_seq for idx, seq in data_filter.items()}

order_keys_loss = sorted(loss_seqs, key=loss_seqs.get)

min_value = loss_seqs[order_keys_loss[0]]
max_value = loss_seqs[order_keys_loss[-1]]

num_scenes   = 8

if best_or_worst == 'best':
  scenes = order_keys_loss[:num_scenes]
else:
  scenes = order_keys_loss[-num_scenes:]

print('Min Loss:', min_value, 'Max Loss:', max_value)
print('------------------------------------------------------------')

# ----------------------------------------------------------------------
# --------------------- PLOTTING VIDEOS --------------------------------

for scene in scenes:
    # ------------------------------------------------------------------
    filename = scene[0][:-3]
    _, _, idx_part, _, idx_movie, _, _, idx_chap, _, idx_scene = filename.split('_')
    movie_name = movies[idx_movie]
    prefix_filename = 'metadata_participant_' + idx_part + '_stimuli_' + idx_movie

    print('Participant:', idx_part, 'Movie:', idx_movie, movie_name, 'Chapter: ', idx_chap, 'Seq:', idx_scene)
    #print('Loss:',loss_seqs[scene])
    # ------------------------------------------------------------------

    stim_metadata = read_csv(datapath + 'data_python/' +  prefix_filename + '_stim.csv')
    stim_initial = stim_metadata.iloc[int(idx_movie)-1]['stimStart']

    avi_filename = datapath + 'stimuli/' + movie_name + '/' + movie_name + '-ch' + str(int(idx_chap)) + '_padded.avi'
    t_ini, t_end = time_sequence(datapath, idx_movie, idx_scene, type_seq)

    # ------------------------------------------------------------------
    rep_video(avi_filename, t_ini, t_end, stim_initial,csvpath + filename + '.csv')

    print('------------------------------------------------------------')    