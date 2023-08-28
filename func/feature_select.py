# Off-line modeling program/ Transfer learning

# Authors: Pan Lincong <panlincong@tju.edu.cn>
# Date: 2023/08/16
# License: MIT License

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from numpy import ndarray
import math

class MutualInformationSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k:int = -1):
        self.k = k

    def fit(self, X: ndarray, y):
        if self.k == -1:
            self.k = math.ceil(0.3 * X.shape[1])
        self.mutual_infos_ = mutual_info_classif(X, y)
        return self
    
    def transform(self, X: ndarray):
        mask = self._get_support_mask()
        return X[:, mask]

    def _get_support_mask(self):
        mask = np.zeros_like(self.mutual_infos_, dtype=bool)
        mask[np.argsort(self.mutual_infos_)[-self.k:]] = True
        return mask

