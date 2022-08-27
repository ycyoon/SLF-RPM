python test.py --gpu 0 --epochs 100 --batch_size 128 --lr 5e-3 --dropout 0 \
	--pretrained "model/ubfc_best.pth" \
	--dataset_name "ubfc-rppg" \
	--dataset_dir "ubfc" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/ubfc/test" \
	--model_depth 18