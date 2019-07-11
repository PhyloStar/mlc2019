import glob
import numpy as np
from collections import defaultdict

for fname in sorted(glob.glob("top20_words_clusters/*")):
    nr_cnt = []
    word_clusts = defaultdict()
    nr_clusts = defaultdict(int)
    for line in open(fname, "r"):
        arr = line.strip().split("\t")
        clust = sorted([arr[0]]+arr[1].split(","))
        if len(clust) >= 2:
            word_clusts[",".join(clust)] = len(clust)
            nr_clusts[str(len(clust))] += 1

    nr_cnt = list(word_clusts.values())

    if len(nr_cnt) < 10: continue

#    print(list(word_clusts.keys())[:10])

#        len_inflecs = len(arr[1].split(","))
#        if len_inflecs > 1:
#            nr_cnt.append(len_inflecs)
#    print(fname.split("/")[-1], np.mean(nr_cnt), np.median(nr_cnt))
    har_mean = [v/float(k) for k, v in nr_clusts.items()]
    wt_mean = [v*float(k) for k, v in nr_clusts.items()]
    denom = [v for k, v in nr_clusts.items()]
    wt_mean = round(sum(wt_mean)/sum(denom), 4)
    har_mean = round(sum(har_mean), 4)
    surface_sim = round(float(fname.split("/")[-1].replace(".words.clusters.txt", "").split("_")[-1]), 3)
    print(fname.split("/")[-1].split("-")[0], wt_mean, surface_sim)
#    print(list(nr_clusts.items()))
