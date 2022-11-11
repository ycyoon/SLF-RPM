python train_tscan.py --epochs 20 --batch_size 32 --lr 1e-3  \
	--trainval_dataset_name "$1" \
	--dataset_dir "data/$1" \
	--workers 8 --vid_frame 180 --vid_frame_stride 1 \
	--log_dir "./logs/$1/tscan" \
	--model_name tscan  
