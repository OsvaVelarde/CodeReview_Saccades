import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

PATH = '/home/osvaldo/Documents/CCNY/Project_Saccades/results/'

OPTS_FOLDER = {'fixed': 'seq_fixed_len',
              'event': 'seq_event_cuts',
              'cont': 'seq_cont_cuts',
              'chapter': 'seq_chapters',
              'wo_rec': 'seq_chapters_wo_recurrence'}

criterias = [['movie'],['movie','chapter']]
per = 0.1 #0.1

# ======================================================================
# ======================================================================

parser = argparse.ArgumentParser(description='Recurrent Feedback for Saccades')

parser.add_argument('--exp', required=True, type=int, help='IDx of experiment')
parser.add_argument('--type-seq', required=True, help='Type of sequence')

args = parser.parse_args()

# --------------------------------------------------------------------
datafolder = PATH + 'predictions/' + OPTS_FOLDER[args.type_seq] + '/'
plotfolder = PATH + 'plots/exp_' + '{:02d}'.format(args.exp) + '/' + OPTS_FOLDER[args.type_seq] + '_loss/'

with open(datafolder + 'exp_' + '{:02d}'.format(args.exp) + '_loss.pkl','rb') as f:
    loss = pkl.load(f)

# --------------------------------------------------------------------

fig1, axs1 = plt.subplots(2,2,figsize=(15,15), sharex=True)

for ax in axs1.flat:
    ax.set_xscale('log')
    ax.set_xlim(1,3000)
    ax.set_yscale('log')

for idx in range(2):
    axs1[1][idx].set_xlabel('Time Step')
    axs1[idx][0].set_ylabel('Loss')


if args.type_seq == 'wo_rec':
    axs1[0][0].set_ylim(0.8,1)
    axs1[0][1].set_ylim(1,1.03)
else:
    axs1[0][0].set_ylim(0.2,0.4)
    axs1[0][1].set_ylim(0.5,1)

# --------------------------------------------------------------------

idx_seq = []
loss_seq = []

for kk, vv in loss.items():
    info_seq  = kk[0][:-3].split('_')
    idx_part  = int(info_seq[2])
    idx_movie = int(info_seq[4])
    idx_chap  = int(info_seq[7])
    idx_scene = 1 if args.type_seq in ['chapter', 'wo_rec'] else int(info_seq[9])
    idx_seq.append([idx_part,idx_movie,idx_chap,idx_scene]) # part - movie - chap - scene
    loss_seq.append(vv)

df_info = pd.DataFrame(idx_seq,columns=['participant','movie','chapter','scene'])
df_loss = pd.DataFrame(loss_seq)
# --------------------------------------------------------------------

n_seq, n_time = df_loss.shape
df_filter = df_loss.dropna(axis=1, thresh=int(per*n_seq))
#df_filter = df_loss.copy()
df = pd.concat([df_info, df_filter], axis=1)

# --------------------------------------------------------------------

mean_data = df_filter.mean(skipna=True)
error_data = df_filter.sem(skipna=True).fillna(0.)
num_samples = df_filter.count()

norm_init_data = df_filter.apply(lambda x: x/x[0], axis=1)
mean_norm_init_data = norm_init_data.mean(skipna=True)
error_norm_init_data = norm_init_data.sem(skipna=True).fillna(0.)

axs1[0][0].plot(mean_data.index + 1, mean_data.values)
axs1[0][1].plot(mean_norm_init_data.index + 1, mean_norm_init_data.values)

axs_samples = axs1[0][0].twinx()
axs_samples.set_ylabel('Samples', color='red')
axs_samples.plot(num_samples.index + 1, num_samples.values, color='red')

#fig.tight_layout()  # otherwise the right y-label is slightly clipped

# --------------------------------------------------------------------

for idx_cc, cc in enumerate(criterias):
    groups = df.groupby(cc)

    for group_idx, df_gg in groups:
        df_gg.drop(columns=['participant','movie','chapter','scene'],inplace=True)
        mean_data = df_gg.mean(skipna=True)
        error_data = df_gg.sem(skipna=True).fillna(0.)
        num_samples = df_gg.count()

        norm_init_data = df_gg.apply(lambda x: x/x[0], axis=1)
        mean_norm_init_data = norm_init_data.mean(skipna=True)
        error_norm_init_data = norm_init_data.sem(skipna=True).fillna(0.)

        # --------------------------------------------------------------------

        if idx_cc == 0:
            axs1[idx_cc+1][0].plot(mean_data.index + 1, mean_data.values,label=group_idx[0])
            axs1[idx_cc+1][1].plot(mean_norm_init_data.index + 1, mean_norm_init_data.values)

            axs1[idx_cc+1][0].legend()
        else:
            fig, axs = plt.subplots(1,2,figsize=(10,5))
            title = 'Movie_' + str(group_idx[0]) + '_Chapter_' + str(group_idx[1]) 
            axs[0].plot(mean_data.index + 1, mean_data.values)
            axs[1].plot(mean_norm_init_data.index + 1, mean_norm_init_data.values)

            axs_samples_gg = axs[0].twinx()
            axs_samples_gg.set_ylabel('Samples', color='red')
            axs_samples_gg.plot(num_samples.index + 1, num_samples.values, color='red')

            for ax in axs.flat:
                ax.set_xscale('log')
                ax.set_yscale('log')

            fig.savefig(plotfolder + 'Loss_Recurrence_' + title + '.svg')
            plt.close(fig)

    print('------------------------')

fig1.savefig(plotfolder + 'Loss_Recurrence_Average.svg')
#plt.show()