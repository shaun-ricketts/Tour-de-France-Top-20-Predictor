from __future__ import annotations

"""tdf_feature_engineer.py

A lightweight, **stateless** feature–engineering transformer that recreates just the
feature‑building logic from your exploratory notebook, focussed on the **`sent`**
scenario that you decided to keep.  Drop this file in your ``/src`` package and
wire it into any scikit‑learn pipeline (or call it directly).

Example
-------
>>> from tdf_feature_engineer import TdfSentFeatureEngineer
>>> engineer = TdfSentFeatureEngineer()  # sent‑inel based imputation
>>> Xt = engineer.fit_transform(prepared_df)  # returns features only
>>> y = engineer.get_target(prepared_df)      # returns the binary target

If you would rather get the full «X + y» DataFrame in a single call, set
``return_target=True`` when calling :py:meth:`transform` or :py:meth:`fit_transform`.
"""
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ["TdfSentFeatureEngineer"]


class TdfSentFeatureEngineer(BaseEstimator, TransformerMixin):
    """Builds the *sent*‑scenario features for Tour‑de‑France modelling.

    The transformer is deliberately *stateless*: it does no learning during
    :py:meth:`fit` beyond validating that the required columns are present.  That
    makes it safe to reuse across cross‑validation folds or production
    inference.

    Parameters
    ----------
    sentinel : int, default ``999``
        Value used to *fill* missing categorical‑numeric ranks such as
        ``"DNF"``/``"DSQ"``.
    required_year_min : int | None, default ``2012``
        Remove rows *before* this (inclusive lower bound).  Passing ``None``
        disables the filter.
    target_top_k : int, default ``20``
        Threshold applied to *TDF_Pos* to create the binary target ``is_top{k}``.
    return_target : bool, default ``False``
        If *True* the rows returned by :py:meth:`transform` **include** the
        target column; otherwise only feature columns are returned.
    """

    #: columns that become the *sent* versions  + flag columns
    _SENT_COLUMNS_: Tuple[str, ...] = (
        "Best_Pos_BT_UWT",
        "Best_Pos_BT_PT",
        "best_recent_tdf_result",
        "best_recent_other_gt_result",
    )

    #: numeric / unchanged columns that form the "core" feature block
    _CORE_COLUMNS_: Tuple[str, ...] = (
        "Age",
        "FC_Pos_YB",
    )

    def __init__(
        self,
        *,
        sentinel: int = 999,
        required_year_min: int | None = 2012,
        target_top_k: int = 20,
        return_target: bool = False,
    ) -> None:
        self.sentinel = sentinel
        self.required_year_min = required_year_min
        self.target_top_k = target_top_k
        self.return_target = return_target

        # will be populated in `fit`
        self.feature_names_: List[str] | None = None
        self.target_name_: str | None = None

    # ------------------------------------------------------------------
    # scikit‑learn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):  # type: ignore[override]
        """Validate columns; nothing to learn."""
        self._validate_columns(X)
        # we *could* compute feature_names_ here, but easier after transform()
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        """Return engineered feature matrix (and optionally the target).

        If *return_target* was *True* when the transformer was instantiated a
        **copy** of the target column is concatenated onto the right‑hand side
        of the feature DataFrame for convenience.
        """
        df = self._prepare_dataframe(X.copy())

        # ------------------------------------------------------------------
        # Assemble final feature matrix
        # ------------------------------------------------------------------
        sent_cols = [f"{col}_sent" for col in self._SENT_COLUMNS_]
        flag_cols = [f"{col}_sent_flag" for col in self._SENT_COLUMNS_]
        feature_cols = list(self._CORE_COLUMNS_) + sent_cols + flag_cols

        self.feature_names_ = feature_cols  # for external inspection
        self.target_name_ = f"is_top{self.target_top_k}"

        if self.return_target:
            return df[feature_cols + [self.target_name_]]
        return df[feature_cols]

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------
    def get_target(self, X: pd.DataFrame) -> pd.Series:
        """Extract the binary target *without* altering *X*.

        Equivalent to ``engineer.transform(X, return_target=True)[target_name]``.
        """
        df = X.copy()
        _, y = self._create_target(df)
        return y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = set(self._CORE_COLUMNS_ + list(self._SENT_COLUMNS_) + [
            "TDF_Pos",
            "Year",
        ]) - set(X.columns)
        if missing:
            raise ValueError(
                f"Input frame missing required columns: {sorted(missing)}"
            )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # ------------------------------------------------------------------
        # Year filtering (optional)
        # ------------------------------------------------------------------
        if self.required_year_min is not None:
            df = df[df["Year"] >= self.required_year_min]

        # ------------------------------------------------------------------
        # Build *_sent and *_flag columns
        # ------------------------------------------------------------------
        for col in self._SENT_COLUMNS_:
            sent_col = f"{col}_sent"
            flag_col = f"{col}_sent_flag"

            df[sent_col] = (
                df[col]
                .replace({"DNF": np.nan, "DSQ": np.nan})
                .fillna(self.sentinel)
                .astype(float)
                .astype(int)
            )
            df[flag_col] = df[col].isnull().astype(int)

        # ------------------------------------------------------------------
        # Target variable and cleaning of TDF_Pos
        # ------------------------------------------------------------------
        df, _ = self._create_target(df)
        return df

    def _create_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # Exclude riders that have no final classification
        mask_valid = ~df["TDF_Pos"].isin(["DNF", "DSQ"])
        df = df.loc[mask_valid].copy()
        df["TDF_Pos"] = pd.to_numeric(df["TDF_Pos"])  # convert str → int

        target_col = f"is_top{self.target_top_k}"
        df[target_col] = (df["TDF_Pos"] <= self.target_top_k).astype(int)
        return df, df[target_col]
