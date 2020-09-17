# CUDA_VISIBLE_DEVICES=0 python myTrain.py -dec=PIN -bsz=32 -dr=0.2 -lr=0.001 -le=1
CUDA_VISIBLE_DEVICES=0 nohup python -u myTrain.py -dec=PIN -bsz=32 -dr=0.3 -lr=0.001 -le=1 > train.log 2>&1 &