import traceback
import sys
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
import torch
from transformers import T5Tokenizer
from unicodedata import category
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k=k)
    min_values = torch.unsqueeze(torch.min(values, -1).values, 1)# values[:, -1, tf.newaxis]
    return torch.where(
        logits < min_values,
        torch.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )

def gather_nd(x, indices):
    newshape = list(indices.shape[:-1] + x.shape[indices.shape[-1]:]) + [1]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([torch.tensor([x.__getitem__(tuple(i))]) for i in indices]).reshape(tuple(newshape))
    return out

def sample_sequence(input_ids, model, max_length, start_token_id, temperature=0.7, top_k=40, no_sample=False):
    batch_size = input_ids.size()[0]
    decoder_input_ids = torch.tensor([start_token_id for _ in range(batch_size)])[:, None].to(device)
    # tokens_embed = model.transformer.get_input_embeddings()
    for _ in range(max_length):
        logits = model(input_ids, decoder_input_ids=decoder_input_ids)["logits"]
        logits = logits[:, -1, :] / temperature

        if no_sample:
            prev = torch.topk(logits, 1)[1]
        else:
            logits = top_k_logits(logits, k=top_k)
            probs = torch.nn.functional.softmax(logits, -1)
            prev = torch.multinomial(probs, 1)
        decoder_input_ids = torch.cat([decoder_input_ids, prev], 1)
    return decoder_input_ids

model_name_path = sys.argv[1]
device = sys.argv[2]
task_name = sys.argv[3]
data_dir = sys.argv[4]
output_file_name = sys.argv[5]
lang = sys.argv[6]
print("using %s"%device)
print(model_name_path)

with open("./%s/test.source"%(data_dir), "r") as fin:
    ipt = [line.strip() for line in fin]
with open("./%s/test.target"%(data_dir), "r") as fin:
    opt = [line.strip() for line in fin]
chrs = (chr(i) for i in range(sys.maxunicode + 1))
punctuation = set(c for c in chrs if category(c).startswith("P"))
def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def pro(token_list, tokenizer, lang):
    string = tokenizer.decode(token_list)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<pad>", "").replace("<s>", "").strip()
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    string = " ".join(string.strip().split())
    if "zh" in lang:
        string = strB2Q(string)
    return string.strip()

tokenizer = T5Tokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id

tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

model = T5ForConditionalGeneration.from_pretrained(model_name_path).to(device)
print("pad:", tokenizer.pad_token_id)
print("start:", model.config.decoder_start_token_id)
print("write to %s"%output_file_name)
with open(output_file_name, "w") as fout:
    batch_size = 16
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
            try:
                if task_name == "st2mo":
                    gen = model.generate(input_ids, do_sample=True, max_length=100, top_k=40, temperature=0.7, decoder_start_token_id=0 if "en" in lang else 1)
                elif task_name == "mo2st":
                    gen = model.generate(input_ids, do_sample=True, max_length=512, top_p=0.9, decoder_start_token_id=0 if "en" in lang else 1)
                else:
                    raise Exception("wrong task name")
            except:
                max_length = 100 if task_name == "st2mo" else 512
                gen = sample_sequence(input_ids=input_ids, model=model, max_length=max_length, temperature=0.7, top_k=40, start_token_id=1 if "zh" in lang else 0)
                traceback.print_exc()
                print("error")
            for ip, op, truth in zip(input_ids, gen, opt[st:ed]):
                print(tokenizer.decode(ip))
                print(tokenizer.decode(op))
                print(truth)
                print("="*10)
                fout.write(pro(op, tokenizer, lang)+"\n")
