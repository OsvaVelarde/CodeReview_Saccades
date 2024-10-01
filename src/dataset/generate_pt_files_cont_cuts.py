import pandas as pd
import torch
import json
import os
import cv2
import math
import naplib as nl
import numpy as np

path_data = '/media/osvaldo/OMV5TB/vinay_data/'
path_sacc = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/data_python/'
path_movie = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/stimuli/'
path_scenes = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/seq_cont_cuts_v2/'

W_PX_MONITOR=1920.0
H_PX_MONITOR=1080.0
PXS_MONITOR = {'x':W_PX_MONITOR ,'y':H_PX_MONITOR}
PADD_ = 100
D = 50

movies = {
    1: 'The_Big_Sick',
    2: 'The_Peanut_Butter_Falcon',
    3: 'Whiplash',
    4: 'Room',
    6: 'Me_Earl_And_Dying_Girl',
    9: 'The_Tomorrow_Man',
    11: 'Dom_Hemingway',
    12: 'Life_After_Beth',
    13: 'Woodshock',
    14: 'The_Comedian'}
# -----------------------------------------------------------

list_filenames = os.listdir(path_data)

# -----------------------------------------------------------

for filename in list_filenames[25:]:
  print('Processing - Participant: ' + filename)

  participant_idx = filename.split('_')[0]
  data = nl.io.load(path_data + filename)

  for dd in data:
    movie_idx = dd['stim']
    chap_idx = dd['ch']
    moviename = movies[movie_idx]
    videoname = moviename + '-ch' + str(chap_idx) + '_padded.avi'

    print('Movie: ' + str(movie_idx) + ' - Chapter: ' + str(chap_idx))

    # -----------------------------------------------------
    try:
      df_stim = pd.read_csv(path_sacc + 'metadata_participant_' + participant_idx + '_stimuli_{:02d}'.format(movie_idx) + '_stim.csv', index_col = 'FileName')
      df_sacc = pd.read_csv(path_sacc + 'metadata_participant_' + participant_idx + '_stimuli_{:02d}'.format(movie_idx) + '_sacc_chap_{:02d}'.format(chap_idx) + '.csv')
      df_sacc.dropna(inplace=True)
    except:
      print('Cannot read csv files')
      continue

    # ----------------------------------------------------

    upper_lim_times = np.where(dd['cont_cut'] == 1)[0]

    if len(upper_lim_times)==0:
      cont_scenes = np.array([[0 , len(dd['cont_cut'])-1]])/100.0
    else:
      lower_lim_times = np.concatenate(([0], upper_lim_times[:-1]))
      cont_scenes = np.column_stack((lower_lim_times, upper_lim_times))/100.0

    # -----------------------------------------------------

    video_path = path_movie + moviename + '/' + videoname
    cap = cv2.VideoCapture(video_path)

    W_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    k_vertical = H_PX_video/H_PX_MONITOR
    k_horizontal = W_PX_video/W_PX_MONITOR

    stimStart_chap = df_stim['stimStart'].loc[videoname]
    videoframerate = df_stim['VideoFrameRate'].loc[videoname]
    # ----------------------------------------------------

    for scene_idx, scene in enumerate(cont_scenes):
      t_i, t_e = scene

      df_sacc_in_scene = df_sacc[(df_sacc['timestamps_rel_stim'] > t_i) & (df_sacc['timestamps_rel_stim_end'] <= t_e)]

      if len(df_sacc_in_scene)<3:
        print('Short Sequence')
        continue

      conds = []
      for kk in ['start','end']:
        for dim in ['x','y']:
          column =  'pos_' + kk + '_pixels_' + dim
          conds.append((df_sacc_in_scene[column]>=0).all() and (df_sacc_in_scene[column]<= PXS_MONITOR[dim]).all())

      if not all(conds):
        print('Removing saccade sequence')
        continue

      df_sacc_in_scene.to_csv(path_scenes + 'csv/' + 'metadata_participant_' + participant_idx + '_stimuli_{:02d}'.format(movie_idx) + '_sacc_chap_{:02d}'.format(chap_idx) + '_scene_' + str(scene_idx) + '.csv')
      list_saccades = []

      for idx_sacc, row_sacc in df_sacc_in_scene.iterrows():
        # -----------------------------------------------
        t_start_sacc = stimStart_chap + row_sacc['timestamps_rel_stim']
        t_end_sacc = stimStart_chap + row_sacc['timestamps_rel_stim_end']
        frame_start_sacc = math.floor(t_start_sacc * videoframerate)
        frame_end_sacc = math.floor(t_end_sacc * videoframerate)

        # -----------------------------------------------
        x_start_sacc  = int(row_sacc['pos_start_pixels_x']*k_horizontal)
        y_start_sacc  = int(row_sacc['pos_start_pixels_y']*k_vertical)
        x_end_sacc    = int(row_sacc['pos_end_pixels_x']*k_horizontal)
        y_end_sacc    = int(row_sacc['pos_end_pixels_y']*k_vertical)

        # -----------------------------------------------
        frames_sacc = [frame_start_sacc,frame_end_sacc]
        center_sacc = [[x_start_sacc,y_start_sacc],[x_end_sacc,y_end_sacc]]
   
        sacc_tensors = []

        for ii in range(2):
          cap.set(cv2.CAP_PROP_POS_FRAMES, frames_sacc[ii])
          ret, frame = cap.read()

          if not ret:
            print("No se pudo leer el frame")
            exit()

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          tensor_frame = torch.tensor(frame_rgb.transpose(2, 0, 1))
          tensor_padded = torch.nn.functional.pad(tensor_frame, (PADD_, PADD_, PADD_, PADD_))

          # ---------------------------------------------
          x_left  = center_sacc[ii][0]+PADD_-D
          x_right = center_sacc[ii][0]+PADD_+D
          y_top   = center_sacc[ii][1]+PADD_-D
          y_bott  = center_sacc[ii][1]+PADD_+D

          patch = tensor_padded[:,y_top:y_bott,x_left:x_right]
          sacc_tensors.append(patch)

          # ---------------------------------------------

        list_saccades.append(sacc_tensors)

      torch.save(list_saccades, path_scenes + 'pts/' + 'metadata_participant_' + participant_idx + '_stimuli_{:02d}'.format(movie_idx) + '_sacc_chap_{:02d}'.format(chap_idx) + '_scene_' + str(scene_idx) + '.pt')

    cap.release()
