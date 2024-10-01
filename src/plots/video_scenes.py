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

type_seq = 'fixed'
moviepath = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/'
csvpath  = moviepath + foldernames[type_seq]+ '/csv/'
datapath = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'

timepath = {'event':datapath,
            'cont':'seq_cont_cuts',
            'fixed':csvpath}

# --------------------------------------------------------------------
# -------------------- SELECTION OF SEQUENCES ------------------------
scenes = ['metadata_participant_07_stimuli_09_sacc_chap_10_seq_3'] 

# ----------------------------------------------------------------------
# --------------------- PLOTTING VIDEOS --------------------------------

for scene in scenes:
    # ------------------------------------------------------------------
    filename = scene
    _, _, idx_part, _, idx_movie, _, _, idx_chap, _, idx_scene = filename.split('_')
    movie_name = movies[idx_movie]
    prefix_filename = 'metadata_participant_' + idx_part + '_stimuli_' + idx_movie

    print('Participant:', idx_part, 'Movie:', idx_movie, movie_name, 'Chapter: ', idx_chap, 'Seq:', idx_scene)
    #print('Loss:',loss_seqs[scene])
    # ------------------------------------------------------------------

    stim_metadata = read_csv(datapath + 'data_python/' +  prefix_filename + '_stim.csv')
    stim_initial = stim_metadata.iloc[int(idx_movie)-1]['stimStart']

    avi_filename = datapath + 'stimuli/' + movie_name + '/' + movie_name + '-ch' + str(int(idx_chap)) + '_padded.avi'
    t_ini, t_end = time_sequence(timepath[type_seq], idx_part, idx_movie, idx_chap, idx_scene, type_seq)

    if type_seq == 'fixed':
      t_ini+=stim_initial
      t_end+=stim_initial

    # ------------------------------------------------------------------
    rep_video(avi_filename, t_ini, t_end, stim_initial,csvpath + filename + '.csv')

    print('------------------------------------------------------------')