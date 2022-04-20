device=cuda:6
task=st2mo
lang=zh
truth_file=./storal_zh_label_test.jsonl
result_file=./st2mo/ra_t5_st2mo_zh.txt
python3 ./eval.py $device $task $lang $truth_file $result_file