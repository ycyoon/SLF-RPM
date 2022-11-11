python train_tscan.py --epochs 100 --batch_size 32 --lr 1e-3  \
	--trainval_dataset_name "$1" \
	--dataset_dir "data/$1" \
	--workers 8 --vid_frame 180 --vid_frame_stride 1 \
	--log_dir "./logs/$1/tscan" \
	--model_name tscan  \
	#--pretrained "logs/$1/pretrain_srfrpm/best_train_model.pth.tar"
	#--retrain "logs/$1/vivit/best_test_model.pth.tar"	
	#--pretrained "log/seq_pretrain/best_train_model_at_20.pth.tar"
	#--evaluate \
	#--retrain "logs/ubfc2/vivit/best_test_model.pth.tar"

