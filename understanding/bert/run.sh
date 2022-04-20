export DATA_PATH=./data
env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u bert.py \
  --save_steps 300 \
  --save_total_limit 50 \
  --model_name_or_path ./model \
  --train_file $DATA_PATH/train.json \
  --validation_file $DATA_PATH/valid.json \
  --test_file $DATA_PATH/test.json \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --max_seq_length 512 \
  --pad_to_max_length \
  --per_device_train_batch_size 10 \
  --learning_rate 3e-5 \
  --num_train_epochs 60 \
  --output_dir ./ours \
  --overwrite_output_dir
