python test.py --gpu 0 --evaluate \
	--pretrained "logs/ubfc/test/best_test_model.pth.tar" \
	--dataset_name "ubfc-rppg" \
	--dataset_dir "data" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/ubfc/test" \
	--model_depth 18
