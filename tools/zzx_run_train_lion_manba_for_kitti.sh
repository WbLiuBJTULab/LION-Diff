#! /bin/bash


## kitti mamba
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
--cfg_file ./cfgs/kitti_models/zzx_second_with_lion_mamba_64dim.yaml \
--extra_tag bs4_zzx_second_with_lion_mamba_64dim \
--batch_size 4  --epochs 4 --max_ckpt_save_num 40 --workers 4 --sync_bn



