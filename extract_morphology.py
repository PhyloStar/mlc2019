from gensim.models import FastText

import sys, utils, glob
from distance_metrics import lcs
import numpy as np

sents = []

surface_sim = 0.8
semantic_sim = 0.6

corpus_names = sorted(glob.glob("PBC49/Texts/*.txt"))

if "train_all" in sys.argv:
    
    for fname in corpus_names:
        print("Processing ", fname)
        sents = []
        out_model = "models/"+fname.split("/")[-1]+".model"
        for line in open(fname, "r"):
            if line.startswith("#"): continue
            arr = line.strip().split("\t")
            sents.append(arr[1].lower().split(" "))

        model_bible = FastText(sents, size=100, window=10, min_count=2, workers=4, iter = 10)
        model_bible.save(out_model)

        model_bible = FastText.load(out_model)
        path_clusters = open(out_model.split("/")[-1].replace(".model", "")+".words.clusters.txt", "w")

        for word in model_bible.wv.vocab:
        #    print(word, model_bible.wv.most_similar(word))
            sim_words = []
            for x in model_bible.wv.most_similar(word):
                if x[1] < semantic_sim: continue
                llcs = lcs.llcs(word, x[0])/max(len(x[0]), len(word))
#                prefix = utils.prefix(word, x[0])
        #        print(word, x[0], llcs)
                if llcs >= surface_sim:# and prefix > 0:
                    sim_words.append(x[0])
            if len(sim_words) > 0:
                print(word, ",".join(sim_words), sep="\t", file=path_clusters)

        path_clusters.close()

if "inflections" in sys.argv:
    for surface_sim in np.arange(0.65, 1.0, 0.05):
#        print(surface_sim)
#        continue
        for fname in corpus_names:
            print("Processing ", fname)

            out_model = "models/"+fname.split("/")[-1]+".model"

            model_bible = FastText.load(out_model)
            path_clusters = open(out_model.split("/")[-1].replace(".model", "")+"_topn_20_"+"lcs_"+str(surface_sim)+".words.clusters.txt", "w")

            for word in model_bible.wv.vocab:
            #    print(word, model_bible.wv.most_similar(word))
                sim_words = []
                for x in model_bible.wv.most_similar(word, topn=20):
    #                if x[1] < semantic_sim: continue
                    llcs = lcs.llcs(word, x[0])/max(len(x[0]), len(word))
    #                prefix = utils.prefix(word, x[0])
            #        print(word, x[0], llcs)
                    if llcs >= surface_sim:# and prefix > 0:
                        sim_words.append(x[0])
                if len(sim_words) > 0:
                    print(word, ",".join(sim_words), sep="\t", file=path_clusters)

            path_clusters.close()
