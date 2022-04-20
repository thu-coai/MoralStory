python3 ./prepare_nltk.py
env CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 python3 -u finetune.py \
    --data_dir ./data_dir \
    --train_name train \
    --output_dir=./output \
    --save_top_k 60 \
    --train_batch_size=10 \
    --eval_batch_size=10 \
    --num_train_epochs 30 \
    --model_name_or_path ./model \
    --learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --n_val 2 \
    --val_check_interval 1.0 \
    --gradient_accumulation_steps 1 \
    --overwrite_output_dir \
    --lang zh