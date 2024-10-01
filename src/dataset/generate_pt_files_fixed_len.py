import pandas as pd
import torch
import json
import os
import cv2
import math

path_data = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'
meta_folder = 'data_python/'
scene_folder = 'scenecuts/'
videos_folder = 'stimuli/'
path_scenes = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/seq_fixed_len/'

W_PX_MONITOR=1920.0
H_PX_MONITOR=1080.0
PXS_MONITOR = {'x':W_PX_MONITOR ,'y':H_PX_MONITOR}
PADD_ = 100
LEN_SEQ = 50
D = 50

# -----------------------------------------------------------

list_filenames = [ff.split('.')[0] for ff in os.listdir(path_data + 'metadata')]

# -----------------------------------------------------------

for filename in list_filenames:
  print('Processing -' + filename)
  movie_idx = filename[-2:]

  try:
    df_stim = pd.read_csv(path_data + meta_folder + filename + '_stim.csv')
    df_chapters = pd.read_csv(path_data + meta_folder + filename + '_chapters.csv')
  except:
    print('Cannot read csv file')
    continue

  df = pd.concat([df_chapters, df_stim], axis=1)

  num_seqs_per_experiment = 0

  for index, row in df.iterrows():
    print('Processing Chapter ' +str(row['chapter_no']))
    # -----------------------------------------------------
    try:
      df_sacc = pd.read_csv(path_data + meta_folder + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '.csv')
    except:
      print('Cannot read chapter')
      continue

    # -----------------------------------------------------
    videoname = row['FileName']
    moviename = videoname.split('-')[0]
    video_path = path_data + videos_folder + moviename + '/' + videoname
    cap = cv2.VideoCapture(video_path)

    W_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    k_vertical = H_PX_video/H_PX_MONITOR
    k_horizontal = W_PX_video/W_PX_MONITOR

    # -----------------------------------------------------
    stimStart_chap = row['stimStart']
    list_dfs = [df_sacc.iloc[i:i + LEN_SEQ] for i in range(0, len(df_sacc), LEN_SEQ)] # LEN_SEQ

    # -----------------------------------------------------
    num_seqs_per_chapters = 0

    for idx_seq, seq in enumerate(list_dfs):

      conds = []
      for kk in ['start','end']:
        for dim in ['x','y']:
          column =  'pos_' + kk + '_pixels_' + dim
          conds.append((seq[column]>=0).all() and (seq[column]<= PXS_MONITOR[dim]).all())

      if not all(conds):
        print('Removing sequence')
        continue

      num_seqs_per_chapters+=1

      seq.to_csv(path_scenes + 'csv/' + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '_seq_' + str(idx_seq) + '.csv')
      list_saccades = []

      for idx_sacc, row_sacc in seq.iterrows():
        t_start_sacc = stimStart_chap + row_sacc['timestamps_rel_stim']
        t_end_sacc = stimStart_chap + row_sacc['timestamps_rel_stim_end']
        frame_start_sacc = math.floor(t_start_sacc * row['VideoFrameRate'])
        frame_end_sacc = math.floor(t_end_sacc * row['VideoFrameRate'])
        # ------------------------------------------------

        x_start_sacc  = int(row_sacc['pos_start_pixels_x']*k_horizontal)
        y_start_sacc  = int(row_sacc['pos_start_pixels_y']*k_vertical)
        x_end_sacc    = int(row_sacc['pos_end_pixels_x']*k_horizontal)
        y_end_sacc    = int(row_sacc['pos_end_pixels_y']*k_vertical)
        # ------------------------------------------------

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

        list_saccades.append(sacc_tensors)

      torch.save(list_saccades, path_scenes + 'pts/' + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '_seq_' + str(idx_seq) + '.pt')
    cap.release()

    print('Num Seqs in the Chapter:',num_seqs_per_chapters)
    print('-----------------------------------------------------')
    num_seqs_per_experiment += num_seqs_per_chapters


  print('Num Seqs in the Experiment:',num_seqs_per_experiment)
  print('===================================================')