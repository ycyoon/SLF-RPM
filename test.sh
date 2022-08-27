#!/bin/bash

# MAHNOB-HCI
python3 test.py --epochs 100 --batch_size 128 --lr 5e-3 \
	--pretrained "./model/mahnob_best.pth" \
	--dataset_name "mahnob-hci" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/mahnob/test" \
	--model_depth 18

# VIPL-HR-V2
python3 test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 \
	--pretrained "./logs/vipl/train/best_train_model.pth.tar" \
	--dataset_name "vipl-hr-v2" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/vipl/test" \
	--model_depth 18

# UBFC
python3 test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 --dropout 0 \
	--pretrained "./logs/ubfc/train/best_train_model.pth.tar" \
	--dataset_name "ubfc-rppg" \
	--dataset_dir "path/to/dataset" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/ubfc/test" \
	--model_depth 18