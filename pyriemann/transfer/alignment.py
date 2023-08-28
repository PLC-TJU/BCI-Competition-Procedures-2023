# Recursive Riemannian/Euclidean Alignment.
#
# Authors: Corey Lin <coreylin2023@outlook.com>
# Date: 2023/06/27
# License: MIT License


from typing import Optional
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pyriemann.utils.covariance import covariances
from pyriemann.utils import mean_covariance
from pyriemann.utils.base import invsqrtm

class Alignment(BaseEstimator, TransformerMixin):
    """Riemannian/Euclidean Alignment."""

    def __init__(
        self,
        align_method: str = "euclid",
        cov_method: str = "lwf",
        n_samples: Optional[int] = 6,
    ):
        self.align_method = align_method
        self.cov_method = cov_method
        self.n_samples = n_samples

    def fit(self, X: ndarray, y: Optional[ndarray] = None):
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        C = covariances(X, estimator=self.cov_method)
        
        self.iC12_ = mean_covariance(C, metric=self.align_method) 
        
        return self

    def transform(self, X):
        if not hasattr(self, "n_tracked"):
            # 如果没有，就给self添加一个n_tracked属性，并赋值为1
            self.n_tracked = 1
            self.temp_iC12_ = self.iC12_
            
        X = np.copy(X)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        #X = X - np.mean(X, axis=-1, keepdims=True)
        C = covariances(X, estimator=self.cov_method)
                    
        for i in range(len(C)):            
            self.n_tracked += 1
            alpha = 1 / (self.n_tracked)
            
            covmats = np.concatenate((self.temp_iC12_[np.newaxis,:], C[i][np.newaxis,:]), axis=0)
            sample_weight = np.array([1-alpha, alpha])
            self.C_ = mean_covariance(covmats, metric=self.align_method, sample_weight=sample_weight) 
            self.temp_iC12_ = invsqrtm(self.C_) 
            
            if (self.n_tracked-1) % self.n_samples == 0: 
                self.iC12_ = self.temp_iC12_

            X[i] = self.iC12_ @ X[i]
        
        return X