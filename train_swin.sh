python train_swin.py --epochs 500 --batch_size 128 --lr 4e-3 --seed 0 \
	--dataset_name "cohface" \
	--dataset_dir "data/cohface" \
	--workers 4 --vid_frame 150 --vid_frame_stride 2 \
	--log_dir "./logs/cohface/swin" \
	--pretrained "/home/yoon/git/SLF-RPM/log/face_pretrain/best_train_model.pth.tar" \
	--finetune head

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
