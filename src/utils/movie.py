import cv2
import pandas as pd

W_PX_MONITOR=1920.0
H_PX_MONITOR=1080.0
PADD_ = 100

# ------------------------------------------------------------------------------

def time_sequence(datapath, type_seq, idx_part, idx_movie, idx_chap, idx_scene=None):
  if type_seq == 'event':
    df_scenecuts = pd.read_excel(datapath + 'scenecuts/T_scenecuts_all_' + idx_movie + '.xlsx' )
    df_scene = df_scenecuts[df_scenecuts['scene_no'] == int(idx_scene)]
    ini_sec = float(df_scene['scene_start_time_padded'].iloc[0])
    end_sec = float(df_scene['scene_end_time_padded'].iloc[0])

  if type_seq == 'fixed':
    df_sacc = pd.read_csv(datapath+'metadata_participant_' + idx_part + '_stimuli_' + idx_movie + '_sacc_chap_' + idx_chap + '_seq_' + idx_scene + '.csv')
    ini_sec = df_sacc.iloc[0]['timestamps_rel_stim']-1
    end_sec = df_sacc.iloc[-1]['timestamps_rel_stim_end']+1

  if type_seq == 'cont':
    ini_sec = 0
    end_sec = 10

  if type_seq == 'chapter':
    ini_sec = 0
    end_sec = 180

  return ini_sec, end_sec

# ------------------------------------------------------------------------------

def rep_video(avifilename, ini_sec, end_sec, stiminitial, saccfilename):
  cap = cv2.VideoCapture(avifilename)
  cap.set(cv2.CAP_PROP_POS_MSEC, ini_sec*1000)

  fps = 60
  delay = int(1000 / fps)
  D = 50

  W_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H_PX_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  k_vertical = H_PX_video/H_PX_MONITOR
  k_horizontal = W_PX_video/W_PX_MONITOR

  dfsacc = pd.read_csv(saccfilename)

  while(cap.isOpened()):
      ret, frame = cap.read()

      time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
      time_rel = time - stiminitial
      row_sacc = dfsacc[(dfsacc['timestamps_rel_stim'] <= time_rel) & (dfsacc['timestamps_rel_stim_end'] >= time_rel)]

      if ret == True:
          if len(row_sacc) == 1: 
            x_start_sacc  = int(row_sacc['pos_start_pixels_x']*k_horizontal)
            y_start_sacc  = int(row_sacc['pos_start_pixels_y']*k_vertical)
            x_end_sacc    = int(row_sacc['pos_end_pixels_x']*k_horizontal)
            y_end_sacc    = int(row_sacc['pos_end_pixels_y']*k_vertical)

            cv2.rectangle(frame, (x_start_sacc-D, y_start_sacc+D), (x_start_sacc+D, y_start_sacc-D), color=(0, 255, 0), thickness=2)
            cv2.rectangle(frame, (x_end_sacc-D, y_end_sacc+D), (x_end_sacc+D, y_end_sacc-D), color=(0, 0, 255), thickness=2)
            cv2.circle(frame, (x_start_sacc, y_start_sacc), radius=5, color=(0, 255, 0), thickness=3)
            cv2.circle(frame, (x_end_sacc, y_end_sacc), radius=5, color=(0, 0, 255), thickness=3)

          cv2.imshow('Frame',frame)

          if time > end_sec:
              break

          if cv2.waitKey(delay) & 0xFF == ord('q'):
              break

      else: 
          break

  cap.release()
  cv2.destroyAllWindows()