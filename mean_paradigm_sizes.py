import glob, sys
import numpy as np
from collections import defaultdict

for threshold in np.arange(0.1, 1.0, 0.05):

    for fname in sorted(glob.glob("top100_clusters/*")):

        nr_cnt = []
        word_clusts = defaultdict()
        nr_clusts = defaultdict(int)
        for line in open(fname, "r"):
            arr = line.strip().split("\t")
            words = [arr[0]]

            if len(arr) < 2: continue

            for x in arr[1].split(","):
                if len(x) < 2: continue

                if "::" not in x: continue

                wrd, score = x.split("::")

                score = float(score.replace(":",""))
                if score >= threshold:
                    words.append(wrd)

            clust = sorted(words)
            if len(clust) >= 2:
                word_clusts[",".join(clust)] = len(clust)
                nr_clusts[str(len(clust))] += 1

        nr_cnt = list(word_clusts.values())

        if len(nr_cnt) < 10: continue

#        har_mean = [v/float(k) for k, v in nr_clusts.items()]
        wt_mean = [v*float(k) for k, v in nr_clusts.items()]
        denom = [v for k, v in nr_clusts.items()]
        wt_mean = round(sum(wt_mean)/sum(denom), 4)
#        har_mean = round(sum(har_mean), 4)
        print(fname.split("/")[-1].split("-")[0], wt_mean, round(threshold, 4), sep="\t")

