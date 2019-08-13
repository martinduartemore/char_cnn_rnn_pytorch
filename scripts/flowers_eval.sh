python3 sje_eval.py \
    --seed 123 \
    --dataset flowers \
    --model_type cvpr \
    --data_dir /A/martin/datasets/flowers_dataset/cvpr2016_flowers \
    --eval_split test \
    --num_txts_eval 0 \
    --print_class_stats True \
    --batch_size 40 \
    --model_path ckpt/sje_flowers_c10_hybrid/sje_flowers_c10_hybrid_0.00070_1_trainval_2019_08_13_14_11_41.pth
