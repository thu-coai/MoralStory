sw = []
for f in ["stopwords", "cn_stopwords", "scu_stopwords", "hit_stopwords", "baidu_stopwords"]:
    with open("%s.txt"%f, encoding="utf-8") as fin:
        print(f)
        for line in fin:
            sw.append(line.strip())
sw = sorted(list(set(sw)))
with open("all_stopwords.txt", "w", encoding="utf-8") as fout:
    fout.write("\n".join(sw))
