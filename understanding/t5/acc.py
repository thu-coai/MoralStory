from logging import log
import sys
import torch
import os
import numpy as np
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
    )

model_name_path = sys.argv[1]
device = sys.argv[2]
data_dir = sys.argv[3]
data_name = sys.argv[4]
lang = sys.argv[5]

with open("./%s/%s.source"%(data_dir, data_name), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/%s.target"%(data_dir, data_name), "r") as fin:
    opt = [line.strip() for line in fin]

tokenizer = AutoTokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)] + ["<mask>"]})
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_path).to(device)
model.eval()

num = 0
batch_size = 24
st, ed = 0, 0
with torch.no_grad():
    while ed < len(ipt):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
        
        input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512)
        tgt_ids = tokenizer(opt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
        decoder_input_ids = model._shift_right(tgt_ids)
        src_ids = input_ids.input_ids.to(device)
        src_mask = input_ids.attention_mask.to(device)
        outputs = model(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, output_hidden_states=True, use_cache=False)

        # [batch_size, length, hidden_size]
        encoder_hidden_states = outputs["decoder_hidden_states"][-1] #outputs["decoder_last_hidden_state"]
        # [batch_size, length]
        mask1 = torch.eq(decoder_input_ids, torch.tensor(tokenizer.convert_tokens_to_ids(["<mask>"])[0]).to(decoder_input_ids.device)).float()
        if lang == "zh":
            bound = 32000
        elif lang == "en":
            bound = 250101
        else:
            raise Exception("language error")        
        mask2 = 1 - torch.lt(decoder_input_ids, bound).float()
        mask_tmp = torch.eq(torch.cumsum(mask1, 1).int(), 0).float()
        mask2 *= mask_tmp

        # [batch_size, length]
        logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
        logits -= (1 - mask2) * (1e20)
        # lprobs = torch.nn.functional.softmax(logits, dim=-1)
        label = torch.sum(tgt_ids * mask1, 1).int()

        for ip, op, truth in zip(decoder_input_ids, logits, label):
            # try:
            pred = op.to("cpu").numpy().tolist()
            id_ = np.argmax(pred)
            pred_id = int(ip[id_].to("cpu").numpy())
            if pred_id == truth:
                num += 1
print("accuracy:", num / len(opt))
