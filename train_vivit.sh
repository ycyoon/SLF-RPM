python train_swin.py --epochs 100 --batch_size 32 --lr 4e-3  \
	--dataset_name "$1" \
	--dataset_dir "data/$1" \
	--workers 8 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/$1/vivit" \
	--finetune head \
	--model_name vivit  \
	--pretrained "logs/$1/pretrain_srfrpm/best_train_model.pth.tar"
	#--retrain "logs/$1/vivit/best_test_model.pth.tar"	
	#--pretrained "log/seq_pretrain/best_train_model_at_20.pth.tar"
	#--evaluate \
	#--retrain "logs/ubfc2/vivit/best_test_model.pth.tar"

#python test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 --dropout 0 \
#	--pretrained "model/ubfc_best.pth" \
#	--dataset_name "ubfc-rppg" \
#	--dataset_dir "data" \
#	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
#	--log_dir "./logs/ubfc/test" \
#	--model_depth 18

#python test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 --dropout 0 \
#	--pretrained "model/mahnob_best.pth" \
#	--dataset_name "mahnob-hci" \
#	--dataset_dir "data2" \
#	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
#	--log_dir "./logs/mahnob/test" \
#	--model_depth 18
