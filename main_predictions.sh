#!/bin/bash

_PATH='./'
PATHDATA='/home/osvaldo/Documents/KirbyPro/home/Movie_Scenes/'
TITLE='exp_'$1
TYPE_SEQ='chapter'

export PYTHONPATH=$(pwd):$PYTHONPATH

# -------------------------------------------------------------
echo 'Inference Stage - Model' $TITLE

python3.8 src/predictions.py \
    --PATH $_PATH \
    --title  $TITLE\
    --datapath $PATHDATA \
	--type-seq $TYPE_SEQ \
	--cfgfilename $_PATH'models/'$TITLE'.cfg' \
    --device-type 'gpu' \
    --gpu-number 0 \
    --resume $_PATH'results/checkpoints/ckpt_'$TITLE'.pth'