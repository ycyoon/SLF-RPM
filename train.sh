python train.py --gpu 0 --epochs 200 --batch_size 128 --lr 5e-4 --dropout 0.3 \
	--dataset_name "merged" \
	--dataset_dir "data" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/merged/test" \
	--model_depth 50 \
	--scratch



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
