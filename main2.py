# Off-line modeling program 2/2

# Authors: Pan Lincong <panlincong@tju.edu.cn>
# Date: 2023/08/16
# License: MIT License

import numpy as np
import pickle, joblib
import os
from functools import partial
from joblib import Parallel, delayed
from func.func import read_blocks, extract_samples_and_labels, downsample_and_extract
from func.func2 import convert_pipeline_model, merge_items, modeling

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

if __name__ == '__main__':
    filepath = './result'
    folder_path = filepath + '/model' 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    subID = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    sublist = []
    for sub in subID:
        sublist.append('TrainData/'+sub)
    Paraset1, Paraset2, Paraset3 = [], [], []
    for id in range(1, 9): 
        file = filepath + '/AccOfSub{}.pkl'.format(id)
        if not os.path.exists(file):
            print("File '{}' does not exist. Skipping...".format(file))
            Paraset1.append([])
            Paraset2.append([])
            Paraset3.append([])
            continue
        with open(file, "rb") as f:
            data = pickle.load(f) 
            combinations = data[0]
            acc = data[-1]
        acc_sub = np.nan_to_num(acc)
        acc_sub = acc_sub.flatten()
        acc_sorted = np.sort(acc_sub, axis=0)[::-1]
        acc_index = np.argsort(acc_sub, axis=0)[::-1]
        combinations_sorted = [combinations[i] for i in acc_index]
        paraset1, _ = zip(*[(x,y) for (x,y) in zip(combinations_sorted,acc_sorted) if x[0][1][1] <= 1.5])
        paraset2, _ = zip(*[(x,y) for (x,y) in zip(combinations_sorted,acc_sorted) if 2 < x[0][1][1] <= 2.5])
        paraset3, _ = zip(*[(x,y) for (x,y) in zip(combinations_sorted,acc_sorted) if 3 < x[0][1][1] <= 3.5])
        Paraset1.append(paraset1)
        Paraset2.append(paraset2)
        Paraset3.append(paraset3)
        
    all_samples, all_labels = [], []
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
      
    for sub in range(0,8):
        paraset1 = Paraset1[sub][:150]
        paraset2 = Paraset2[sub][:150]
        paraset3 = Paraset3[sub][:150]
        paraset1, _ = merge_items(paraset1)
        paraset2, _ = merge_items(paraset2)
        paraset3, _ = merge_items(paraset3)
        targetData, targetLabel = all_samples[sub], all_labels[sub]
        sourceData = [x for i, x in enumerate(all_samples) if i != sub]
        sourceLabel = [x for i, x in enumerate(all_labels) if i != sub]
        paraset = paraset1 + paraset2 + paraset3
        func = partial(modeling, targetData=targetData, targetLabel=targetLabel, sourceData=sourceData, sourceLabel=sourceLabel)
        Model = Parallel(n_jobs=-1, verbose=len(paraset))(delayed(func)(param) for param in paraset)
        Model = [item for sublist in Model for item in sublist]
        Model = [x for x in Model if x]
        Model1 = [model for model in Model if model.para[1]<=2]
        Model2 = [model for model in Model if 2<model.para[1]<=3]
        Model3 = [model for model in Model if 3<model.para[1]<=4]
        Model1.sort(key=lambda x: x.acc, reverse=True)
        Model2.sort(key=lambda x: x.acc, reverse=True)
        Model3.sort(key=lambda x: x.acc, reverse=True)
        Models = Model1 + Model2 + Model3
        for Model in Models:
            convert_pipeline_model(Model.model, type = 16)
        count1 = len(Model1)
        count2 = len(Model2)
        count3 = len(Model3)
        models_ind1=range(0,count1)
        models_ind2=range(count1,count1+count2)
        models_ind3=range(count1+count2,count1+count2+count3)
        models_ind = [models_ind1,models_ind2,models_ind3]
        Models.append(models_ind)
        joblib.dump(Models, folder_path + '/modelforperson{}.joblib'.format(sub+1), compress=9)
        
