from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#Object to binarize categorical data
class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse = True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.encoded_data = self.label_encoder.fit_transform(X.ravel())
        self.encoded_data = self.one_hot_encoder.fit_transform(self.encoded_data.reshape(-1,1))
        return self.encoded_data
