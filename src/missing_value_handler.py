from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FillWithSentinel(BaseEstimator, TransformerMixin):
    """
    Replaces 'DNF', 'DSQ', and NaN with 999 in all columns
    that start with 'Best_' or 'best_'.
    """
    def __init__(self, sentinel=999):
        self.sentinel = sentinel
        self.columns_to_clean = []

    def fit(self, X, y=None):
        # Identify columns to clean
        self.columns_to_clean = [
            col for col in X.columns if col.startswith('Best_') or col.startswith('best_')
        ]
        return self

    def transform(self, X):
        X = X.copy()
        # Replace 'DNF', 'DSQ' and NaN with the sentinel value
        X[self.columns_to_clean] = (
            X[self.columns_to_clean]
            .replace(['DNF', 'DSQ'], self.sentinel)
            .fillna(self.sentinel)
        )
        return X
