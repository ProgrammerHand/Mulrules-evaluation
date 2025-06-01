import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scalers_dict, continuous_col_names, continuous_cols):
        self.scalers_dict = scalers_dict
        self.continuous_col_names = continuous_col_names
        self.continuous_cols = continuous_cols
        self.col_index_to_name = {idx: name for idx, name in zip(self.continuous_cols, self.continuous_col_names)}

    def fit(self, X, y=None):
        return self  # No fitting needed here if already fitted outside

    def transform(self, X):
        return self._apply_scalers(X, method='transform')

    def inverse_transform(self, X):
        return self._apply_scalers(X, method='inverse_transform')

    def _apply_scalers(self, X, method='transform'):
        temp = X.copy()
        if isinstance(temp, np.ndarray):
            for col in self.continuous_cols:
                scaler = self.scalers_dict[self.col_index_to_name[col]]
                temp[:, col] = getattr(scaler, method)(temp[:, col].reshape(-1, 1)).flatten()
        elif isinstance(temp, pd.DataFrame):
            for col in self.continuous_col_names:
                scaler = self.scalers_dict[col]
                temp[col] = getattr(scaler, method)(temp[[col]])
        else:
            temp = np.array(temp)
            for col in self.continuous_cols:
                scaler = self.scalers_dict[self.col_index_to_name[col]]
                temp[:, col] = getattr(scaler, method)(temp[:, col].reshape(-1, 1)).flatten()
        return temp