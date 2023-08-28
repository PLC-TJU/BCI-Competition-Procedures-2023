# Off-line modeling program 1/2

# Authors: Pan Lincong <panlincong@tju.edu.cn>
# Date: 2023/08/16
# License: MIT License

import os, pickle, joblib
import numpy as np
from joblib import Parallel, delayed
import itertools
from pyriemann.estimation import Covariances
from pyriemann.channelselection import ElectrodeSelection, FlatChannelRemover
from pyriemann.spatialfilters import ElectrodeSelection_CSPCS
from func.func import read_blocks, extract_samples_and_labels, downsample_and_extract
from func.func2 import merge_items_backwards, calc_acc_post
      
if __name__ == '__main__':
    folder_path = 'result' 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    subID = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    sublist = []
    for sub in subID:
        sublist.append('TrainData/'+sub)
    all_samples, all_labels, all_ch = [], [], []
    for sub in sublist:
        dataAll, srate, channels = read_blocks(sub)
        samples, labels = [], []
        for data in dataAll:
            sample, label = extract_samples_and_labels(data, srate)
            samples.append(sample)
            labels.append(label)
        samples = np.concatenate(samples, axis=0)   
        labels = np.concatenate(labels, axis=0)  
        del data, dataAll  
        samples = downsample_and_extract(samples, fs_old=1000, fs_new=250, window=(0,4))
        samples = np.float32(samples)
        all_samples.append(samples)
        all_labels.append(labels)
        cs = FlatChannelRemover()
        all_ch.append(cs.fit(samples).channels_)

    chlist = [list(range(0, 59)), list(range(16, 59))]
    timeset = [[0,1.5],[0.5,1.5],[0.5,2.5],[0.5,3.5]]
    freqset=[[1,8],[1,12],[4,12],[4,20],[5,30],[8,26],
             [8,32],[10,26],[10,32],[12,26],[12,30],
             [18,26],[18,32],[22,32],[26,32],[26,40]]
    file = 'allpara36new.joblib'
    _, _, algorlist=joblib.load(file)
    paraset = list(itertools.product(timeset, freqset))
    paraset = [sublist1 + sublist2 for sublist1, sublist2 in paraset]
    
    for sub in range(0,8):
        samples, labels = all_samples[sub], all_labels[sub]
        covs = Covariances(estimator='lwf').transform(samples[:,all_ch[sub],:])
        sel_ch1 = ElectrodeSelection_CSPCS(nelec=28).fit(covs, labels).subelec_
        sel_ch2 = ElectrodeSelection(nelec=28).fit(covs, labels).subelec_
        chanset = [all_ch[sub][sel_ch1],all_ch[sub][sel_ch2]] + chlist
        temp = set(tuple(x) for x in chanset)
        chanset = [list(x) for x in temp]
        combinations = list(itertools.product(chanset, paraset))
        combinations = list(itertools.product(combinations, algorlist))
        Acc = Parallel(n_jobs=-1, verbose=len(combinations))(delayed(calc_acc_post)(combo, sub, all_samples, all_labels) for combo in combinations)
        combinations = merge_items_backwards(combinations)
        Acctemp = [item for sublist in Acc for item in sublist]
        Acc = Acctemp
        for i in range(len(combinations)):
            temp = list(combinations[i])
            temp[2] = Acc[i]
            combinations[i] = tuple(temp)           
        f = open(folder_path + '/AccOfSub{}.pkl'.format(sub+1), 'wb')
        pickle.dump((combinations, Acc), f)
        f.close()


