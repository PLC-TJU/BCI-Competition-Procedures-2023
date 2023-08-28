# 2023 BCI Competition MI Dataset Preprocessing

# Authors: Pan Lincong <panlincong@tju.edu.cn>
# Date: 2023/08/16
# License: MIT License

import numpy as np
from typing import List, Tuple
from collections import defaultdict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from pyriemann.transfer import TLCenter, TLStretch, TLRotate, TLClassifier
from pyriemann.classification import FgMDM
from sklearn.model_selection import KFold
from sklearn.base import clone
from func.func import split_eeg
from func.algorithmslist2 import Basemodel

def merge_items(paraset: List[Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int],
                List[int]], float]]) -> List[Tuple[Tuple[List[int], List[int]], Tuple[List[int], 
                List[int], List[int]], float]]:  
    grouped = defaultdict(list)
    for index, item in enumerate(paraset):
        key = tuple(tuple(x) for x in item[0])
        grouped[key].append((index, item))
    result = []
    indices = []
    for key, items in grouped.items():
        first = tuple(list(x) for x in key)
        second0 = [list(x) for x in zip(*[x[1][1] for x in items])]
        second = [[item[0] for item in sublist] for sublist in second0]
        third = [x[1][2] for x in items]
        result.append((first, second, third))
        indices.extend([x[0] for x in items])
    return result, indices


def merge_items_backwards(result: Tuple[List[Tuple[Tuple[List[int], List[int]], Tuple[List[List[int]], 
                List[List[int]], List[List[int]]], List[float]]], List[int]], indices = None) -> List[
                Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int], List[int]], float]]:
    paraset = []
    for item in result:
        first = item[0]
        second = item[1]
        third = item[2]
        for i in range(len(second[0])):
            new_second = tuple([[second[j][i]] for j in range(len(second))])
            paraset.append((first, new_second, third[i]))
    if indices is not None:
        paraset = [x for _, x in sorted(zip(indices, paraset), key=lambda pair: pair[0])]
    return paraset

def convert_pipeline_model(model, type = 32):
    if isinstance(model, Pipeline):
        for name, est in model.named_steps.items():
            if type == 16:
                convert_sklearn_model16(est)
            elif type == 32:
                convert_sklearn_model32(est)
            else:
                convert_sklearn_model64(est)

def convert_sklearn_model32(model):
    if isinstance(model, LDA):
        if hasattr(model, 'covariance_') and isinstance(model.covariance_, np.ndarray) and model.covariance_.dtype != np.float32:
            model.covariance_ = model.covariance_.astype(np.float32)
        if hasattr(model, 'scalings_') and isinstance(model.scalings_, np.ndarray) and model.scalings_.dtype != np.float32:
            model.scalings_ = model.scalings_.astype(np.float32)
    elif isinstance(model, SVC):
        if hasattr(model, 'support_vectors_') and isinstance(model.support_vectors_, np.ndarray) and model.support_vectors_.dtype != np.float32:
            model.support_vectors_ = model.support_vectors_.astype(np.float32)
    elif isinstance(model, TLCenter):
        model.recenter_ = {k: v for k, v in model.recenter_.items() if k == model.target_domain}
    elif isinstance(model, TLStretch):
        model._means = {k: v for k, v in model._means.items() if k == model.target_domain}
    elif isinstance(model, TLRotate):
        model.rotations_ = None
    elif isinstance(model, FgMDM):
        if hasattr(model._fgda, '_W') and isinstance(model._fgda._W, np.ndarray) and model._fgda._W.dtype != np.float32:
            model._fgda._W = model._fgda._W.astype(np.float32)
        model._fgda._lda = None
    elif isinstance(model, TLClassifier):
        convert_pipeline_model(model.estimator, type = 32)

def convert_sklearn_model64(model):
    if isinstance(model, LDA):
        if hasattr(model, 'covariance_') and isinstance(model.covariance_, np.ndarray) and model.covariance_.dtype != np.float64:
            model.covariance_ = model.covariance_.astype(np.float64)
        if hasattr(model, 'scalings_') and isinstance(model.scalings_, np.ndarray) and model.scalings_.dtype != np.float64:
            model.scalings_ = model.scalings_.astype(np.float64)
    elif isinstance(model, SVC):
        if hasattr(model, 'support_vectors_') and isinstance(model.support_vectors_, np.ndarray) and model.support_vectors_.dtype != np.float64:
            model.support_vectors_ = model.support_vectors_.astype(np.float64)
    elif isinstance(model, TLCenter):
        model.recenter_ = {k: v for k, v in model.recenter_.items() if k == model.target_domain}
    elif isinstance(model, TLStretch):
        model._means = {k: v for k, v in model._means.items() if k == model.target_domain}
    elif isinstance(model, TLRotate):
        model.rotations_ = None
    elif isinstance(model, FgMDM):
        if hasattr(model._fgda, '_W') and isinstance(model._fgda._W, np.ndarray) and model._fgda._W.dtype != np.float64:
            model._fgda._W = model._fgda._W.astype(np.float64)
    elif isinstance(model, TLClassifier):
        convert_pipeline_model(model.estimator, type = 64)

def convert_sklearn_model16(model):
    if isinstance(model, LDA):
        if hasattr(model, 'covariance_') and isinstance(model.covariance_, np.ndarray) and model.covariance_.dtype != np.float16:
            model.covariance_ = model.covariance_.astype(np.float16)
        if hasattr(model, 'scalings_') and isinstance(model.scalings_, np.ndarray) and model.scalings_.dtype != np.float16:
            model.scalings_ = model.scalings_.astype(np.float16)
    elif isinstance(model, SVC):
        if hasattr(model, 'support_vectors_') and isinstance(model.support_vectors_, np.ndarray) and model.support_vectors_.dtype != np.float16:
            model.support_vectors_ = model.support_vectors_.astype(np.float16)
    elif isinstance(model, TLCenter):
        model.recenter_ = {k: v for k, v in model.recenter_.items() if k == model.target_domain}
    elif isinstance(model, TLStretch):
        model._means = {k: v for k, v in model._means.items() if k == model.target_domain}
    elif isinstance(model, TLRotate):
        model.rotations_ = None
    elif isinstance(model, FgMDM):
        if hasattr(model._fgda, '_W') and isinstance(model._fgda._W, np.ndarray) and model._fgda._W.dtype != np.float16:
            model._fgda._W = model._fgda._W.astype(np.float16)
        model._fgda._lda = None
    elif isinstance(model, TLClassifier):
        convert_pipeline_model(model.estimator, type = 16)
        
def modeling(combination, targetData, targetLabel, sourceData, sourceLabel):
    ch = combination[0][0]
    para = combination[0][1]
    algor = combination[1]
    if len(combination) > 2:
        acc = combination[2]
    else:
        acc = [0] * len(combination[1][0])
    algorind_NoTL = [[sublist[i] for i in range(len(sublist)) if algor[0][i] == 0] for sublist in algor]
    algorind_TL = [[sublist[i] for i in range(len(sublist)) if algor[0][i] != 0] for sublist in algor]
    indices_NoTL = [i for i in range(len(algor[0])) if algor[0][i] == 0]
    indices_TL = [i for i in range(len(algor[0])) if algor[0][i] != 0]
    indices = indices_NoTL + indices_TL
    models =[]
    basemodel = Basemodel(channel=ch, para=para)
    if not all(not sublist for sublist in algorind_NoTL):
        Models1 = Basemodel(channel=ch, para=para, algorind=algorind_NoTL)
        if para[1] <= 2:
            window_width = 2
        elif para[1] <= 3:
            window_width = 3
        else:
            window_width = 4
        targetData, targetLabel = split_eeg(targetData, targetLabel, fs=250, window_width=window_width, window_step=0.1)
        Models1.fit(targetData,targetLabel)
        for i in range(len(Models1.model)):
            if Models1.info[i]['exception']:
                models.append(None)
            else:
                model = clone(basemodel)
                model.algorind = [[algorind_NoTL[0][i]], [algorind_NoTL[1][i]], [algorind_NoTL[2][i]]]
                model.model = Models1.model[i]
                model.info = Models1.info[i]
                model.acc = acc[indices_NoTL[i]]
                models.append(model)
    if not all(not sublist for sublist in algorind_TL):
        Models2 = Basemodel(channel=ch, para=para, algorind=algorind_TL) 
        Models2.fit(targetData,targetLabel,sourceData,sourceLabel)
        for i in range(len(Models2.model)):
            if Models2.info[i]['exception']:
                models.append(None)
            else:
                model = clone(basemodel)
                model.algorind = [[algorind_TL[0][i]], [algorind_TL[1][i]], [algorind_TL[2][i]]]
                model.model = Models2.model[i]
                model.info = Models2.info[i]
                model.acc = acc[indices_TL[i]]
                models.append(model)
    sorted_models = [models[i] for i in indices]
    return sorted_models

def calc_acc_post(combination, targetsub, all_samples, all_labels):  
    targetData = all_samples[targetsub]
    targetLabel = all_labels[targetsub]
    targetData = np.float32(targetData)
    sourceData = [x for i, x in enumerate(all_samples) if i != targetsub]
    sourceLabel = [x for i, x in enumerate(all_labels) if i != targetsub]
    sourceData = np.float32(sourceData)
    del all_samples, all_labels
    kf = KFold(n_splits=3, shuffle=True)
    Acc = []
    for i in range(3):
        for train_index, test_index in kf.split(targetData):
            targetData_train, targetData_test = targetData[train_index], targetData[test_index]
            targetLabel_train, targetLabel_test = targetLabel[train_index], targetLabel[test_index]
            models = modeling(combination, targetData_train, targetLabel_train, sourceData, sourceLabel)
            accs = []
            for model in models: 
                if model is not None:                 
                    p_labels = model.predict(targetData_test)
                    acc = np.mean(p_labels == targetLabel_test)
                else:
                    acc = 0
                accs.append(acc)
            Acc.append(accs)   
    Acc = np.mean(np.array(Acc), axis=0)      
    return Acc 