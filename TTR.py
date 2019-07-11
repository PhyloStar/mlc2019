import sys, utils, glob

from collections import defaultdict

corpus_names = sorted(glob.glob("PBC49/Texts/*.txt"))

for fname in corpus_names:
    d = defaultdict(int)
    for line in open(fname, "r"):
        if line.startswith("#"): continue
        arr = line.strip().split("\t")
        for word in arr[1].lower().split(" "):
            d[word] += 1

    nr_types, nr_tokens = len(d), sum(list(d.values()))
    print(fname.split("/")[-1].split("-")[0], nr_types, nr_tokens, round(nr_tokens/nr_types,4))
