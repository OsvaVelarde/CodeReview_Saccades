import pandas as pd
import os
import naplib as nl
import numpy as np
import pickle as pkl
import argparse

path_data = '/media/osvaldo/OMV5TB/vinay_data/'
path_sacc = '/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/seq_chapters/csv/'
path_pred = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/predictions/'

parser = argparse.ArgumentParser(description='Recurrent Feedback for Saccades')

parser.add_argument('--exp', required=True, type=int, help='IDx of experiment')
parser.add_argument('--type-seq', required=True, help='Type of sequence', choices=['event','cont'])
parser.add_argument('--remove-rec', action='store_true')
args = parser.parse_args()

# ==================================================================================

folder = 'seq_chapters_wo_recurrence' if args.remove_rec else 'seq_chapters'

with open(path_pred + folder + '/exp_' + '{:02d}'.format(args.exp) + '_loss.pkl','rb') as f:
    loss = pkl.load(f)

list_filenames = os.listdir(path_data)

# ==================================================================================

result_alignment = {}

for filename in list_filenames:
  print('Processing:', filename)
  idx_part = filename.split('_')[0]
  data = nl.io.load(path_data + filename)

  for dd in data:
    idx_movie = dd['stim']
    idx_chap = dd['ch']
    kk = 'metadata_participant_' + idx_part + '_stimuli_' + '{:02d}'.format(idx_movie)  + '_sacc_chap_' + '{:02d}'.format(idx_chap)

    try:
      df_sacc = pd.read_csv(path_sacc + kk + '.csv', usecols = ['timestamps_rel_stim','timestamps_rel_stim_end'])
    except:
      continue

    df_sacc = df_sacc.iloc[3:].reset_index(drop=True)

    try:
      df_loss = pd.DataFrame({'Time': pd.Series(df_sacc.values.flatten('F')), 
                      'Loss': loss[(kk + '.pt',)][1:]})
    except:
      continue

    # ------------------------------------------------
    cuts = dd[args.type_seq + '_cut']

    upper_lim_times = np.where(cuts == 1)[0]

    if len(upper_lim_times)==0:
      scenes = np.array([[0 , len(cuts)-1]])/100.0
    else:
      lower_lim_times = np.concatenate(([0], upper_lim_times[:-1]))
      scenes = np.column_stack((lower_lim_times, upper_lim_times))/100.0

    # ------------------------------------------------
    for scene_idx, scene in enumerate(scenes):
      t_i, t_e = scene

      df_loss_in_scene = df_loss[(df_loss['Time'] > t_i) & (df_loss['Time'] <= t_e)]

      if len(df_loss_in_scene) == 0:
        continue

      result_alignment[kk + '_scene_' + str(scene_idx)] = df_loss_in_scene['Loss'].tolist()

resfilename = path_pred + 'alignments/exp_' + '{:02d}'.format(args.exp) + '_' + args.type_seq + '_loss'
resfilename = resfilename + '_wo_rec.pkl' if args.remove_rec else resfilename + '.pkl'

with open(resfilename, 'wb') as f:
    pkl.dump(result_alignment, f)