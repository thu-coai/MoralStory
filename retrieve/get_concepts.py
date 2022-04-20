#!/usr/bin/env python
import json
import numpy as np
np.random.seed(42)
import jieba 
import jieba.posseg as pseg
jieba.enable_paddle()
from zhon.hanzi import punctuation
import string
allpunct = string.punctuation + punctuation
import spacy
nlp = spacy.load('en_core_web_sm')

stpwrdpath = "./stopwords/all_stopwords.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

def get_concepts(text, lang):
    '''text is a moral '''
    if lang == "zh":
        words = pseg.cut(text.strip(), use_paddle=True)
        filtered = [w for w, pos in words if pos.lower() in ["n", "v", "vd", "vn", "a", "ad", "an", "d"] and w not in stpwrdlst and w not in allpunct]
        return filtered
    elif lang == "en":
        toklist = nlp(text.strip())
        toks = []
        for t in toklist:
            if t.is_stop:
                continue
            if t.pos_.lower() not in ["adj", "adv", "noun", "verb"]:
                continue
            else:
                toks.append(t.lemma_)
        return toks
