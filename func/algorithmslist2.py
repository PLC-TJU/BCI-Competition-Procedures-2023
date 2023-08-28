# Off-line modeling program/ Transfer learning

# Authors: Pan Lincong <panlincong@tju.edu.cn>
# Date: 2023/08/16
# License: MIT License

import numpy as np
from numpy import ndarray
from copy import deepcopy
from scipy.signal import  iircomb, butter, filtfilt
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace as TS
from pyriemann.spatialfilters import CSP
from pyriemann.classification import MDM, FgMDM
from pyriemann.transfer import encode_domains, TLDummy, TLCenter, TLStretch, TLRotate, TLClassifier
from decomposition import DCPM, TRCA
from func.feature_select import MutualInformationSelector as MIBIF
import warnings
warnings.filterwarnings('ignore')

def Modeling_Framework(Xt, yt, Xs = None, ys = None, method_flag = [[0,0],[0,0],[0,1]]):
    if any(x != 0 for x in method_flag[0]):
        DataAll = np.concatenate([np.concatenate(Xs, axis=0), Xt], axis=0)
        LabelAll = np.concatenate(ys + [yt])
        domain = np.concatenate([i * np.ones(len(y), dtype=int) for i, y in enumerate(ys + [yt])])
        DataAll, LabelAll = encode_domains(DataAll, LabelAll, domain)
        source_names = ['{}'.format(i) for i in range(len(Xs))]
        domain_weight = {name: 1 for name in source_names}
        target_domain = str(domain[-1])
        domain_weight.update({target_domain:3})
        total = sum(domain_weight.values())
        for key in domain_weight:
            domain_weight[key] /= total
    if 0 in method_flag[0]:
        DataAll0 = np.float32(Xt)
        LabelAll0 = np.array(yt)
        domain = np.zeros(LabelAll0.shape, dtype=int)
        target_domain = domain[-1] 
    prealignments = {
        0: Pipeline(steps=[]),
        1: make_pipeline(
            Covariances(estimator='lwf'),
            TLDummy(),
        ),
        2: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain, metric='euclid'),
        ),
        3: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain, metric='riemann'),
        ),
        4: make_pipeline(
            Covariances(estimator='lwf'),
            TLCenter(target_domain=target_domain),
            TLStretch(
                target_domain=target_domain,
                final_dispersion=1,
                centered_data=True,
            ),
            TLRotate(target_domain=target_domain, metric='euclid'),
        )
    }
    if not all(i in prealignments for i in method_flag[0]):
        raise ValueError("Invalid value for method_flag[0]: Index of data pre-alignment methods.")
    transformers = {
        0: [CSP(nfilter=6)],
        1: [MDM()],
        2: [FgMDM()],
        3: [TS(), MIBIF()],
        4: [DCPM()],
        5: [TRCA()]
    }
    if not all(i in transformers for i in method_flag[1]):
        raise ValueError("Invalid value for method_flag[1]: Index of feature extraction methods.")
    classifiers = {
        0: [SVC()],
        1: [LDA(solver='eigen', shrinkage='auto')]
    }
    if not all(i in classifiers for i in method_flag[2]):
        raise ValueError("Invalid value for method_flag[2]: Index of classification methods.")
    method_type = {}
    for i, item in enumerate(method_flag[0]):
        key = item
        if key not in method_type:
            method_type[key] = []
        temp = []
        for j in range(len(method_flag)):
            temp.append(method_flag[j][i])
        method_type[key].append(temp)
    method_type = list(method_type.items())    
    Models, Infos = [], []
    for i in range(len(method_type)):
        pa_index = method_type[i][0]
        if method_type[i][0] != 0:
            pa = prealignments[pa_index] 
            pa_base = clone(pa)
            transformers_ = deepcopy(transformers)
            try:
                tempDataAll = pa_base.fit_transform(DataAll, LabelAll)
                exception = False
            except:
                exception = True
        else:
            transformers_ = deepcopy(transformers)
            transformers_[0].insert(0, Covariances(estimator='cov'))
            for n in range(1, 4):
                transformers_[n].insert(0, Covariances(estimator='lwf'))
            exception = False
        for j in range(len(method_type[i][1])):
            fee_index, clf_index = method_type[i][1][j][1], method_type[i][1][j][2]
            if exception:
                   Models.append(None)
            else:
                fee = make_pipeline(*transformers_[fee_index])
                clf = make_pipeline(*classifiers[clf_index])
                estimator = clone(fee)
                estimator.steps += clone(clf).steps
                if method_type[i][0] == 0:
                    try:
                        Model = clone(estimator)
                        Model.fit(DataAll0, LabelAll0)
                        Models.append(Model)
                        exception = False
                    except:
                        Models.append(None)
                        exception = True
                else:
                    clf_fur = make_pipeline(
                        TLClassifier(
                            target_domain=target_domain,
                            estimator=estimator,
                            domain_weight=domain_weight,
                        ),
                    )
                    try:
                        clf_fur_base = clone(clf_fur)
                        clf_fur_base.fit(tempDataAll, LabelAll)
                        steps = pa_base.steps + clf_fur_base.steps
                        Model = Pipeline(steps)
                        Models.append(Model)
                        exception = False
                    except:
                        Models.append(None)
                        exception = True
            Infos.append({
                'exception': exception
            })     
    return Models, Infos

def classify(model,channel,para,X):
    fs = 250
    X = np.reshape(X, (-1, *X.shape[-2:]))
    X_ch = X[:,channel,:]
    start = para[0]
    end = para[1]
    X_cut = cut_data(X_ch, fs, start, end)
    lowcut = para[2]
    highcut = para[3]
    X_filt = butter_bandpass_filter(X_cut, lowcut, highcut, fs, order=5) 
    pred = model.predict(X_filt)
    return pred

def get_pre_filter(data, fs=250):
    f0 = 50
    q = 35
    b, a = iircomb(f0, q, ftype='notch', fs=fs)
    filter_data = filtfilt(b, a, data)
    return filter_data

def cut_data(data, fs, start, end):
    _, _, n_points = data.shape
    t = np.arange(n_points) / fs 
    idx = np.logical_and(t >= start, t < end) 
    data_new = data[:, :, idx] 
    return data_new

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    data = get_pre_filter(data) 
    nyq = 0.5 * fs 
    low = lowcut / nyq 
    high = highcut / nyq 
    b, a = butter(order, [low, high], btype='band')
    data_filtered = filtfilt(b, a, data) 
    return data_filtered

class Basemodel(BaseEstimator, ClassifierMixin):
    def __init__(self, channel=range(16,59), para=[0,2,5,32], algorind=[[0],[0],[0]], **kwds):
        self.channel = channel
        self.para = para
        self.algorind = algorind
        self.kwds = kwds
        self.fs = 250

    def fit(self, Xt: ndarray, yt: ndarray, Xs = None, ys = None):
        fs = self.fs
        if Xs is None:
            X = Xt
            flag = False
        elif isinstance(Xs, list):
            X = []
            X.append(Xt)
            X = X + Xs
            del Xt, Xs
            flag = True
        elif Xs.ndim == 4:
            Xs_list = [Xs[i] for i in range(len(Xs))]
            X = []
            X.append(Xt)
            X = X + Xs_list
            del Xt, Xs
            flag = True
        elif Xs.ndim == 3:
            Xs = np.reshape(Xs, (-1, *Xs.shape[-3:]))
            X = np.concatenate((Xt, Xs), axis=0)
            del Xt, Xs
            flag = True
        if not flag: 
            if any(x != 0 for x in self.algorind[0]):
                raise ValueError(
                    "Transfer learning algorithms are not allowed when there is no input from the source dataset!")   
            else:
                Xt = np.reshape(Xt, (-1, *Xt.shape[-2:]))
                Xt_ch = Xt[:,self.channel,:]
                start = self.para[0]
                end = self.para[1]
                Xt_cut = cut_data(Xt_ch, fs, start, end)
                lowcut = self.para[2]
                highcut = self.para[3]
                Xt_filt = butter_bandpass_filter(Xt_cut, lowcut, highcut, fs, order=5)
                self.model,self.info = Modeling_Framework(Xt_filt, yt, method_flag = self.algorind)
        else:
            ch = self.channel
            start = self.para[0]
            end = self.para[1]
            lowcut = self.para[2]
            highcut = self.para[3]
            all_samples = []
            for samples in X:
                samples = cut_data(samples[:,ch,:], fs, start, end)
                samples = butter_bandpass_filter(samples, lowcut, highcut, fs, order=5)
                all_samples.append(samples)
            sourceData = np.float32(all_samples[1:])
            targetData = np.float32(all_samples[0])
            del all_samples
            self.model,self.info = Modeling_Framework(targetData, yt, sourceData, ys, method_flag = self.algorind)
        return self

    def predict(self, X):
        fs = self.fs
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X_ch = X[:,self.channel,:]
        start = self.para[0]
        end = self.para[1]
        X_cut = cut_data(X_ch, fs, start, end)
        lowcut = self.para[2]
        highcut = self.para[3]
        X_filt = butter_bandpass_filter(X_cut, lowcut, highcut, fs, order=5)         
        if isinstance(self.model, list):            
            pred = []
            for model, info in zip(self.model, self.info):
                if not info['exception']:
                    pred.append(model.predict(X_filt))
                else:
                    pred.append(np.zeros(len(X_filt), dtype=int))
        else:
            if not self.info['exception']:
                pred = self.model.predict(X_filt)
            else:
                pred = np.zeros(len(X_filt), dtype=int)
        return pred
    
    