import json
import sys
import nltk
import numpy as np
from nltk import ngrams
import os
from bert_score import BERTScorer
import copy
from itertools import combinations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge import Rouge
import os
from nltk import data
import jieba
import json
data.path.append(os.environ["HOME"]+"/nltk_data")

def string2token(string, lang):
    if lang == "en":
        return nltk.word_tokenize(string)
    elif lang == "zh":
        return [j for j in jieba.cut(string)]
    else:
        raise Exception("language specification error")

def bleu(ref_token, cand_token):
    result = {}
    for i in range(1, 5):
        result["bleu-%d"%i] = []
        for r, c in zip(ref_token, cand_token):
            result["bleu-%d"%i].append(nltk.translate.bleu_score.sentence_bleu([r], c, weights=tuple([1./i for j in range(i)])))
        result["bleu-%d"%i] = "%.4f"%(np.mean(result["bleu-%d"%i]))
    return result

def bertscore(refs, cands, ipts, device, lang):
    scorer1 = BERTScorer(lang=lang, device=device)
    scorer2 = BERTScorer(lang=lang, device=device, rescale_with_baseline=True, idf=True, idf_sents=ipts+cands)
    with torch.no_grad():
        p1, r1, f11 = scorer1.score(cands=cands, refs=refs)
        p2, r2, f12 = scorer2.score(cands=cands, refs=refs)
    p1, r1, f11 = p1.cpu().numpy().tolist(), r1.cpu().numpy().tolist(), f11.cpu().numpy().tolist()
    p2, r2, f12 = p2.cpu().numpy().tolist(), r2.cpu().numpy().tolist(), f12.cpu().numpy().tolist()
    return {
        "bertscore-p": "%.4f"%(np.mean(p1).item()),
        "bertscore-r": "%.4f"%(np.mean(r1).item()),
        "bertscore-f1": "%.4f"%(np.mean(f11).item()),
        "bertscore-rescale-p": "%.4f"%(np.mean(p2).item()),
        "bertscore-rescale-r": "%.4f"%(np.mean(r2).item()),
        "bertscore-rescale-f1": "%.4f"%(np.mean(f12).item()),
    }

def repetition_distinct(name, cand_token):
    result = {}
    for i in range(1, 5):
        num, all_ngram, all_ngram_num = 0, {}, 0.
        for k, cand in enumerate(cand_token):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
            for s in set(ngs):
                if ngs.count(s) > 1:
                    num += 1
                    break
        result["repetition-%d"%i] = "%.4f"%(num / float(len(cand_token)))
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    return result


def length(name, cand_token):
    length = []
    for c in cand_token:
        length.append(len(c))
    return {"length": "%.4f"%np.mean(length)}

def tokenize(string):
    newstr = ""
    for s in string:
        if s in [",", ".", '"', "?", "'", ";", ":", "!", "…", "-"]:
            newstr += " " + s + " "
        else:
            newstr += s
    return newstr

def rouge(name, ipt, cand):
    rouge_name = "rouge-l"
    item_name = "r"

    res = {}
    res["cover"] = []
    for k, (tmp_ipt, tmp_cand) in enumerate(zip(ipt, cand)):
        for tmp_ref in tmp_ipt.split("#")[2:]:
            tmp_ref = tmp_ref.lower()
            tmp_cand = tmp_cand.lower()
            if lang == "zh":
                tmp_ref = " ".join([w for w in "".join(tmp_ref.strip().split())])
                tmp_hyp = " ".join([w for w in "".join(tmp_cand.strip().split())])
            elif lang == "en":
                tmp_ref = " ".join(tokenize(tmp_ref.strip()).split()).strip()
                tmp_hyp = " ".join(tokenize(tmp_cand.strip()).split()).strip()
            try:
                tmp_res = Rouge().get_scores(refs=tmp_ref, hyps=tmp_hyp)[0]
            except:
                continue
            res["cover"].append(tmp_res[rouge_name][item_name])
    # for name1 in rouge_name:
    #     for name2 in item_name:                
    #         res["%s-%s"%(name1, name2)] = np.mean(res["%s-%s"%(name1, name2)])
    res["cover"] = np.mean(res["cover"])
    return res


def LCS(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def Recon_LCS(x, y, exclusive=True):
    """
    Returns the Longest Subsequence between x and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns:
      sequence: LCS of x and y
    """
    i, j = len(x), len(y)
    table = LCS(x, y)

    def _recon(i, j):
        """private recon calculation"""
        if i == 0 or j == 0:
            return []
        elif x[i - 1] == y[j - 1]:
            return _recon(i - 1, j - 1) + [(x[i - 1], i)]
        elif table[i - 1, j] > table[i, j - 1]:
            return _recon(i - 1, j)
        else:
            return _recon(i, j - 1)

    recon_list = list(map(lambda x: x[0], _recon(i, j)))
    if len(recon_list):
        return "".join(recon_list).strip()
    else:
        return ""
    # return Ngrams(recon_list, exclusive=exclusive)
    # return recon_tuple


def lcs3_dp(input_x, input_y):
    dp = [([0] * (len(input_y)+1)) for i in range(len(input_x)+1)]
    maxlen = maxindex = 0
    for i in range(1, len(input_x)+1):
        for j in range(1, len(input_y)+1):
            if i == 0 or j == 0:
                    dp[i][j] = 0
            if input_x[i-1] == input_y[j-1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    maxindex = i - maxlen
            else:
                dp[i][j] = 0
    return input_x[maxindex:maxindex + maxlen]

def inversenum(a):
    num = 0
    all_num = 0
    for i in range(0,len(a)):
        for j in range(i,len(a)):
            if a[i] > a[j]:
                num += 1
            all_num += 1
    return num / float(all_num)

def find_all(sub,s):
	index_list = []
	index = s.find(sub)
	while index != -1:
		index_list.append(index)
		index = s.find(sub,index+1)
	
	if len(index_list) > 0:
		return index_list
	else:
		return -1

def order(name, ipt, cand, kw2id):
    num = []
    for k, (tmp_ipt, tmp_cand, tmp_kw2id) in enumerate(zip(ipt, cand, kw2id)):
        tmp_ipt, tmp_cand = tmp_ipt.lower(), tmp_cand.lower()
        pos = []
        kw_list = list(tmp_kw2id.keys())
        kw_list.reverse()

        for tmp_ref in kw_list:
            if lang == "zh":
                tmp_ref = "".join(tmp_ref.strip().split())
                tmp_hyp = "".join(tmp_cand.strip().split())
            else:
                tmp_ref = " ".join(tokenize(tmp_ref.strip()).split())
                tmp_hyp = " ".join(tokenize(tmp_cand.strip()).split())                
            lcs = lcs3_dp(tmp_ref, tmp_hyp)
            if len("".join(lcs.strip().split()).strip())>(1 if lang=="zh" else 3):
                pos.append(tmp_hyp.find(lcs))
            else:
                pos.append(-1)
        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])

        new_rank = [-1 for _ in idlist]
        for idl, ord in zip(idlist, orderlist):
            new_rank[idl] = tmp_kw2id[kw_list[ord]]
        num.append(1-inversenum(new_rank))
    return {"order": np.mean(num)}

if __name__ == '__main__':
    device = sys.argv[1]
    task = sys.argv[2]
    lang = sys.argv[3]
    truth_file = sys.argv[4]
    result_file = sys.argv[5]
    # with open("%s/%s_data/data_%s/storal_%s_label_test.json"%(data_dir, task, lang, lang), encoding="utf-8") as fin:
    with open(truth_file, encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
        if task == "mo2st":
            ipt = [d["input"].strip() for d in data]
            truth = [d["story"].strip() for d in data]        

            kw2id = []
            for i1, t1 in zip(ipt, truth):
                i1, t1 = i1.lower(), t1.lower()
                kw_list = i1.strip().split("#")[2:]
                pos = [t1.strip().find(kw.strip()) for kw in kw_list]

                idlist = list(range(len(pos)))
                orderlist = sorted(idlist, key=lambda x: pos[x])
                kw2id.append({})
                for idl, ord in zip(idlist, orderlist):
                    kw2id[-1][kw_list[ord]] = idl
        elif task == "st2mo":
            ipt = [d["story"].strip() for d in data]
            truth = [d["moral"].strip() for d in data]
        else:
            raise Exception("wrong task name!")

    result_list = [result_file]

    all_result = {}

    for name in result_list:
        with open(name, "r", encoding="utf-8") as fin:
            cand = [line.strip() for line in fin]
        eval_result = {}

        ipt_token = [string2token(c.lower(), lang) for c in ipt]
        cand_token = [string2token(c.lower(), lang) for c in cand]
        truth_token = [string2token(t.lower(), lang) for t in truth]
        eval_result.update(bertscore(refs=truth, cands=cand, ipts=ipt, device=device, lang=lang))
        eval_result.update(length(name=name, cand_token=cand_token))
        eval_result.update(bleu(ref_token=truth_token, cand_token=cand_token))
        eval_result.update(repetition_distinct(name=name, cand_token=cand_token))
        if task == "mo2st":
            eval_result.update(rouge(name=name, ipt=ipt, cand=cand))
            eval_result.update(order(name=name, ipt=ipt, cand=cand, kw2id=kw2id))

        model_name = name
        if model_name not in all_result:
            all_result[model_name] = {}

        for key in sorted(eval_result):
            all_result[model_name][key] = eval_result[key]
            print(model_name, key, eval_result[key])
        print("="*10)

    # with open("%s/result_%s.json"%(task, lang), "w") as fout:
    #     json.dump(all_result, fout, indent=4, ensure_ascii=False)
