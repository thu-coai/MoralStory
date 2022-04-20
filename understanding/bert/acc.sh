model_name_path=./model
device=cuda:0
data_dir=./data
python3 ./acc.py  $model_name_path $device $data_dir valid
python3 ./acc.py  $model_name_path $device $data_dir test