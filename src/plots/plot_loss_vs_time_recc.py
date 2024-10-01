import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
epochs = [0,1,2,3,4,5,6,7] #0,50,99
exp = 4
criterias = [['movie'],['movie','chapter']]

for idx, ee in enumerate(epochs):
    fig_epoch, ax_epoch = plt.subplots(3,2,figsize=(10,15))

    for idx in range(2):
        for idy in range(3):
            #ax_epoch[idy][idx].set_xlim(0,200)
            ax_epoch[idy][idx].legend()

    with open('/home/osvaldo/Documents/CCNY/Project_Saccades/results/loss_recurrence/exp_' + '{:02d}'.format(exp) + '/test_epoch_'+ str(ee) + '.pkl','rb') as f:
        data = pkl.load(f)

    idx_scenes = []
    seq_scenes = []

    for kk, vv in data.items():
        info_scene = kk[0][:-3].split('_')
        idx_scenes.append([int(info_scene[2]),int(info_scene[4]),int(info_scene[7]),int(info_scene[9])]) # part - movie - chap - scene
        seq_scenes.append(vv)

    df_info = pd.DataFrame(idx_scenes,columns=['participant','movie','chapter','scene'])
    df_seq = pd.DataFrame(seq_scenes)
    df = pd.concat([df_info, df_seq], axis=1)

    # --------------------------------------------------------------------
    df_gral = df.drop(columns=['participant','movie','chapter','scene'])
    n_seq, n_time = df_gral.shape

    df_filter = df_gral.dropna(axis=1, thresh=int(0.1*n_seq))
    # df_filter = df_gral

    mean_data = df_filter.mean(skipna=True)
    error_data = df_filter.sem(skipna=True).fillna(0.)
    norm_init_data = df_filter.apply(lambda x: x/x[0], axis=1)
    mean_norm_init_data = norm_init_data.mean(skipna=True)
    error_norm_init_data = norm_init_data.sem(skipna=True).fillna(0.)

    mean_rolling = mean_data.rolling(window=3).mean()

    ax_epoch[0][0].plot(mean_data.index, mean_data.values)
    ax_epoch[0][0].plot(error_data.index, mean_data.values-error_data.values,color='black',alpha=0.3)
    ax_epoch[0][0].plot(error_data.index, mean_data.values+error_data.values,color='black',alpha=0.3)
    #ax_epoch[0][0].plot(mean_data.index, mean_rolling.values,color='red')

    ax_epoch[0][1].plot(mean_norm_init_data.index, mean_norm_init_data.values)
    ax_epoch[0][1].plot(error_data.index, mean_norm_init_data.values-error_norm_init_data.values,color='black',alpha=0.3)
    ax_epoch[0][1].plot(error_data.index, mean_norm_init_data.values+error_norm_init_data.values,color='black',alpha=0.3)

    # --------------------------------------------------------------------

    #ax_epoch[0][0].set_ylim(0.1,0.4)
    #ax_epoch[0][0].set_yscale('log')
    #ax_epoch[0][0].set_xscale('log')
    #ax_epoch[0][1].set_ylim(0,4)

    # --------------------------------------------------------------------

    for idx_cc, cc in enumerate(criterias):
        print(cc)
        groups = df.groupby(cc)

        for group_idx, df_gg in groups:
            group_idx = group_idx[0]
            df_gg.drop(columns=['participant','movie','chapter','scene'],inplace=True)
            n_seq, n_time = df_gg.shape
            mean_data = df_gg.mean(skipna=True)
            error_data = df_gg.sem(skipna=True).fillna(0.)
            norm_init_data = df_gg.apply(lambda x: x/x[0], axis=1)
            mean_norm_init_data = norm_init_data.mean(skipna=True)
            error_norm_init_data = norm_init_data.sem(skipna=True).fillna(0.)

            # --------------------------------------------------------------------

            ax_epoch[idx_cc+1][0].plot(mean_data.index, mean_data.values,label=group_idx)
            ax_epoch[idx_cc+1][1].plot(mean_norm_init_data.index, mean_norm_init_data.values)

            ax_epoch[idx_cc+1][0].set_xscale('log')
            ax_epoch[idx_cc+1][0].set_yscale('log')
            ax_epoch[idx_cc+1][1].set_xscale('log')
            ax_epoch[idx_cc+1][1].set_yscale('log')

        print('------------------------')

    for idx in range(2):
        for idy in range(3):
            ax_epoch[idy][idx].legend()


plt.show()



#     df.T.plot(ax=ax[idx][0],legend=False)
#     mean_data.plot(ax=ax[idx][1],legend=False)
#     norm_init_data.T.plot(ax=ax[idx][2],legend=False)

#     ax[idx][1].plot(mean_data.index, mean_data.values)
#     ax[idx][1].fill_between(mean_data.index, mean_data.values - error_data.values, 
#         mean_data.values + error_data.values, color='gray', alpha=0.2)

#     ax[idx][3].plot(mean_norm_init_data.index, mean_norm_init_data.values)
#     ax[idx][3].fill_between(mean_norm_init_data.index, mean_norm_init_data.values - error_norm_init_data.values, 
#         mean_norm_init_data.values + error_norm_init_data.values, color='gray', alpha=0.2)

#     # --------------------------------------------------------------------

#     for kk in range(4):
#         ax[idx][kk].set_xlim(0,300)
#         #ax[idx][kk].set_ylim(0.3,0.4)

# #     ax[idx][1].errorbar(x=np.arange(num_points),y=norm_mean[:num_points])
# #     #ax[1].errorbar(x=np.arange(99),y=mean_data/np.abs(mean_data[0]))

# plt.show()