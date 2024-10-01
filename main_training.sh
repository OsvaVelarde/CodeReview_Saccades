#!/bin/bash

_PATH='./'
#PATHDATA='/media/osvaldo/OMV5TB/Movie_Scenes/pts/'
PATHDATA='/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/'
TITLE='exp_'$1

export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------------------------------------
echo 'Training Stage - Model' $TITLE

python3.8 src/training_v2.py \
    --PATH $_PATH\
    --title  $TITLE\
    --datapath $PATHDATA\
    --cfgfilename $_PATH'models/'$TITLE'.cfg'\
    --device-type 'gpu'\
    --gpu-number 0\
    --print-freq 10