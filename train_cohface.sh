python test.py --gpu 0 --epochs 200 --batch_size 128 --lr 1e-5 --dropout 0.3 \
	--scratch \
	--dataset_name "cohface" \
	--dataset_dir "cohface" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/cohface/test" \
	--model_depth 50 \
    --finetune all