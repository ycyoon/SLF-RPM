python test.py --gpu 0 --epochs 100 --batch_size 256 --lr 5e-4 --dropout 0.3 \
	--pretrained "model/mahnob_best.pth" \
	--dataset_name "mahnob-hci" \
	--dataset_dir "mahnob" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/mahnob/test" \
	--model_depth 18 \
    --finetune all
