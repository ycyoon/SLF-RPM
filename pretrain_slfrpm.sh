python pretrain.py --gpu $2 --epochs 300 --batch_size 64 --lr 1e-4 \
                --n_dim 2048 --temperature 1 \
                --dataset_name $1 \
                --dataset_dir "/home/yoon/data/PPG/preprocess/$1" \
                --workers 4 --vid_frame 150 --clip_frame 75 --roi_list 0 1 2 3 4 5 6 --stride_list 1 2 3 4 5 \
                --log_dir "./logs/$1/pretrain_srfrpm_swin"

