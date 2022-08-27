python main.py --gpu 0 --epochs 300 --batch_size 128 --lr 1e-4 \
		--n_dim 2048 --temperature 1 \
		--dataset_name "merged" \
		--dataset_dir "data" \
		--workers 4 --vid_frame 150 --clip_frame 30 --roi_list 0 1 2 3 4 5 6 --stride_list 1 2 3 4 5 \
		--log_dir "./logs/merged/train" \
		--model_depth 50

python test.py --gpu 0 --epochs 200 --batch_size 128 --lr 5e-4 --dropout 0.3 \
	--pretrained "./logs/merged/train/best_train_model.pth.tar" \
	--dataset_name "merged" \
	--dataset_dir "data" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/merged/test" \
	--model_depth 50 \
    --finetune all



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
