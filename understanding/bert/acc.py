from logging import log
import json
import traceback
import sys
import torch
import numpy as np
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,    
    )

model_name_path = sys.argv[1]
device = sys.argv[2]
data_dir = sys.argv[3]
data_name = sys.argv[4]

with open("./%s/%s.json"%(data_dir, data_name), "r") as fin:
    data = json.load(fin)["data"]
    ipt = [d["text"] for d in data]
    opt = [d["label"] for d in data]

tokenizer = AutoTokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
tokenizer.add_special_tokens({"additional_special_tokens": ["[P]"]})
sp_id = tokenizer.convert_tokens_to_ids("[P]")

model = AutoModelForSequenceClassification.from_pretrained(model_name_path).to(device)
model.eval()

num = []
batch_size = 24
st, ed = 0, 0
all_loss = []
all_logits = []
with torch.no_grad():
    while ed < len(ipt):
        st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
        input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        # [batch_size, length, hidden_size]
        encoder_hidden_states = outputs.hidden_states[-1]
        # [batch_size, length]
        mask1 = torch.eq(input_ids, torch.tensor(sp_id).to(input_ids.device)).float()
        mask2 = torch.eq(input_ids, torch.tensor(tokenizer.mask_token_id).to(input_ids.device)).float()
        logits = torch.sum(torch.matmul(encoder_hidden_states*mask1[:, :, None], torch.transpose(encoder_hidden_states*mask2[:, :, None], 1, 2)), 1)
        logits -= (1 - mask2) * (1e20)
        for idx_, (ip, op, mk, truth) in enumerate(zip(input_ids, logits, mask2, opt[st:ed])):
            pred = op.to("cpu").numpy().tolist()
            id_ = torch.cumsum(mk, 0).to("cpu").numpy().tolist()
            pred_id = int(id_[np.argmax(pred)])
            label_id = int(truth)
            if pred_id == label_id:
                num.append(1)
            else:
                num.append(0)
print("accuracy:", np.sum(num) / len(opt))