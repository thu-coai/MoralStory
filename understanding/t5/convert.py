dir_ = "model_base_post_zh"

import os
f_list = []
for _, _, fl in os.walk(dir_):
    for f in fl:
        if f.startswith("val_avg_loss="):
            f_list.append(f)
    break
map_f = {}
for f in f_list:
    map_f[f.split("=")[2]] = f
for f_name in sorted(map_f):
    f = map_f[f_name]
    print("processing %s"%f)
    os.system("python3 convert_pl_checkpoint_to_hf.py %s/%s ./%s/best_tfmr %s/%s"%(dir_, f, dir_, dir_, f.split(".ckp")[0]))
