model_name_path=./model
device=cuda:0
data_dir=./data
lang=zh
python3 ./acc.py  $model_name_path $device $data_dir val $lang
python3 ./acc.py  $model_name_path $device $data_dir test $lang