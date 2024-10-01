import pandas as pd
import torch
import torch
import json
import os
import cv2
import math

path_data = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'
meta_folder = 'data_python/'
scene_folder = 'scenecuts/'
videos_folder = 'stimuli/'
coeff_px_vda_file = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/coeff_px_vda.json'
path_scenes = '/media/osvaldo/OMV5TB/Movie_Scenes/'

W_PX_MONITOR=1920.0
H_PX_MONITOR=1080.0
PXS_MONITOR = {'x':W_PX_MONITOR ,'y':H_PX_MONITOR}
PADD_ = 100
# -----------------------------------------------------------

list_filenames = [ff.split('.')[0] for ff in os.listdir(path_data + 'metadata')]

# with open(coeff_px_vda_file, "r") as jsonfile:
#     coeffs = json.load(jsonfile)

# -----------------------------------------------------------

for filename in list_filenames[0:20]:
  print('Processing -' + filename)
  movie_idx = filename[-2:]

  try:
    df_stim = pd.read_csv(path_data + meta_folder + filename + '_stim.csv')
    df_chapters = pd.read_csv(path_data + meta_folder + filename + '_chapters.csv')
  except:
    print('Cannot read csv file')
    continue

  df = pd.concat([df_chapters, df_stim], axis=1)

  scenecuts = pd.read_excel(path_data + scene_folder + 'T_scenecuts_all_' + movie_idx + '.xlsx',
                            usecols=['chapter','scene_no','scene_start_time_padded', 'scene_end_time_padded'])

  scenes_per_chapter = scenecuts.groupby(by='chapter')

  # with open(path_data + meta_folder + filename + '_stim.json') as f:
  #   json_stim = json.load(f)

  for index, row in df.iterrows():
    print('Chapter: ' +str(row['chapter_no']))
    # -----------------------------------------------------
    try:
      df_sacc = pd.read_csv(path_data + meta_folder + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '.csv')
    except:
      print('Cannot read chapter')
      continue

    # -----------------------------------------------------
    # try:
    #   slope_x, intercept_x, slope_y, intercept_y = coeffs[filename][str(row['chapter_no'])]
    # except:
    #   print('There is not information to vda-pixel relation for this file')
    #   continue

    # patch_w = (5 - intercept_x)/slope_x
    # patch_h = (5 - intercept_y)/slope_y

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

    # -----------------------------------------------
    df_scenes = scenes_per_chapter.get_group(int(row['chapter_no']))
    df_scenes.reset_index(drop=True,inplace=True)

    for idx_scene, row_scene in df_scenes.iterrows():
      scene_no = row_scene['scene_no'] 
      t_i = row_scene['scene_start_time_padded']
      t_e = row_scene['scene_end_time_padded']

      print('Scene: ' + str(scene_no))

      df_sacc_in_scene = df_sacc[(df_sacc['timestamps_rel_stim'] >= t_i) & (df_sacc['timestamps_rel_stim_end'] <= t_e)]

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

      df_sacc_in_scene.to_csv(path_scenes + 'csv/' + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '_scene_' + str(int(scene_no)) + '.csv')
    
      list_saccades = []

      # -----------------------------------------------

      for idx_sacc, row_sacc in df_sacc_in_scene.iterrows():

        # -----------------------------------------------
        t_start_sacc = stimStart_chap + row_sacc['timestamps_rel_stim']
        t_end_sacc = stimStart_chap + row_sacc['timestamps_rel_stim_end']
        frame_start_sacc = math.floor(t_start_sacc * row['VideoFrameRate'])
        frame_end_sacc = math.floor(t_end_sacc * row['VideoFrameRate'])

        # -----------------------------------------------
        x_start_sacc  = int(row_sacc['pos_start_pixels_x']*k_horizontal)
        y_start_sacc  = int(row_sacc['pos_start_pixels_y']*k_vertical)
        x_end_sacc    = int(row_sacc['pos_end_pixels_x']*k_horizontal)
        y_end_sacc    = int(row_sacc['pos_end_pixels_y']*k_vertical)

        #dx = int(patch_w * k_horizontal/4)
        #dy = int(patch_h * k_vertical/4)
        #D = min(dx,dy)
        D = 50

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

      torch.save(list_saccades, path_scenes + 'pts/' + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '_scene_' + str(int(scene_no)) + '.pt')
    cap.release()



      # # -----------------------------------------------
      # import matplotlib.pyplot as plt 
      # import matplotlib.patches as patches

      # fig, axs = plt.subplots(1,2)

      # # patch_sta = patches.Rectangle((x_start_sacc - dx / 2, y_start_sacc - dy / 2), dx, dy, linewidth=1, edgecolor='r', facecolor='none')
      # # patch_end = patches.Rectangle((x_end_sacc - dx / 2, y_end_sacc - dy / 2), dx, dy, linewidth=1, edgecolor='g', facecolor='none')
      # patch_sta = patches.Rectangle((x_start_sacc - D / 2, y_start_sacc - D / 2), D, D, linewidth=1, edgecolor='r', facecolor='none')
      # patch_end = patches.Rectangle((x_end_sacc - D / 2, y_end_sacc - D / 2), D, D, linewidth=1, edgecolor='g', facecolor='none')

      # patch_plot = [patch_sta,patch_end]

      # axs[0].add_patch(patch_sta)
      # axs[1].add_patch(patch_end)

      # ii = 0

      # for idx_frame in frames_plot:
      #   cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
      #   ret, frame = cap.read()
      #   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      #   axs[ii].imshow(frame_rgb)
      #   axs[ii].plot(sacc_plot[0][0],sacc_plot[0][1],'ro')
      #   axs[ii].plot(sacc_plot[1][0],sacc_plot[1][1],'go')

      #   axs[ii].axis('off')
      #   ii=ii+1
      # break



