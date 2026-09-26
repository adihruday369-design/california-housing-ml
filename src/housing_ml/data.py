"""Download and validate the public California Housing dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn California Housing dataset (downloads on first use)."""
    bunch = fetch_california_housing(as_frame=True, download_if_missing=True)
    frame = bunch.data.copy()
    frame.columns = [str(name) for name in frame.columns]
    target = bunch.target.rename("MedHouseVal")
    if frame.isna().any().any() or target.isna().any():
        raise ValueError("Unexpected missing values in the source dataset")
    return frame, target
