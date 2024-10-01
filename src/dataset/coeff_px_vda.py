import pandas as pd
import matplotlib.pyplot as plt 
import os
from scipy.stats import linregress
import json

path_data = '/media/osvaldo/Seagate Basic/Movie_Saccade_Data/'
meta_folder = 'data_python/'
videos_folder = 'stimuli/'

W_PX_MONITOR=1920.0
H_PX_MONITOR=1080.0

# -----------------------------------------------------------

list_filenames = [ff.split('.')[0] for ff in os.listdir(path_data + 'metadata')]

coeff = {}

for filename in list_filenames:
  print('Processing -' + filename)

  try:
    df_stim = pd.read_csv(path_data + meta_folder + filename + '_stim.csv')
    df_chapters = pd.read_csv(path_data + meta_folder + filename + '_chapters.csv')
  except:
    print('No se pudo ' + filename)
    continue

  df = pd.concat([df_chapters, df_stim], axis=1)
  coeff[filename]={}

  for index, row in df.iterrows():
    try:
      df_sacc = pd.read_csv(path_data + meta_folder + filename + '_sacc_chap_{:02d}'.format(row['chapter_no']) + '.csv')
    except:
      print('No se pudo ' + filename + ' chap' + str(row['chapter_no']))
      continue

    for column in ['pos_start_pixels_x','pos_end_pixels_x']:
      df_sacc = df_sacc.loc[(df_sacc[column] >= 0) & (df_sacc[column] < W_PX_MONITOR)]

    for column in ['pos_start_pixels_y','pos_end_pixels_y']:
      df_sacc = df_sacc.loc[(df_sacc[column] >= 0) & (df_sacc[column] < H_PX_MONITOR)]

    x_px = pd.concat([df_sacc['pos_start_pixels_x'], df_sacc['pos_end_pixels_x']])
    y_px = pd.concat([df_sacc['pos_start_pixels_y'], df_sacc['pos_end_pixels_y']])

    x_vda = pd.concat([df_sacc['pos_start_vda_x'], df_sacc['pos_end_vda_x']])
    y_vda = pd.concat([df_sacc['pos_start_vda_y'], df_sacc['pos_end_vda_y']])

    if len(x_px) < 2:
      continue

    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(x_px, x_vda)
    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(y_px, y_vda)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(x_px,x_vda,'*r')
    # ax.plot(y_px,y_vda,'*b')
    # plt.show()

    coeff[filename][row['chapter_no']] = [slope_x, intercept_x, slope_y, intercept_y]

    print(filename,row['chapter_no'], slope_x, intercept_x, slope_y, intercept_y)

with open('/home/osvaldo/Documents/CCNY/Project_Saccades/results/coeff_px_vda.json', "w") as jsonfile:
    json.dump(coeff, jsonfile)