#!/usr/bin/env python

from transformers import (
    AutoModel,
    AutoTokenizer,
)
import torch

def get_representation(text, model_path="roberta-base-chinese", device="cuda:0"):
    '''text is a string (a story or a moral) '''
    model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        txt_ids = tokenizer([text], return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        txt_outputs = model(txt_ids, output_hidden_states=True)
        txt_rep = torch.mean(txt_outputs.last_hidden_state[0], 0).cpu().numpy().tolist()
    return txt_rep
