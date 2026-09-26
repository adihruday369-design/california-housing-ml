"""Task labels derived from the continuous target using train-only thresholds."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["Value", "Mid-market", "Premium"]


def tier_edges(y_train: pd.Series) -> np.ndarray:
    """Compute tertile cut points from training data only."""
    edges = np.asarray(y_train.quantile([1 / 3, 2 / 3]), dtype=float)
    if not np.isfinite(edges).all() or edges[0] >= edges[1]:
        raise ValueError("Unable to derive distinct price-tier boundaries")
    return edges


def assign_tiers(y: pd.Series, edges: np.ndarray) -> pd.Series:
    """Map target values to fixed, ordered market segments."""
    return pd.Series(
        pd.cut(y, bins=[-np.inf, *edges, np.inf], labels=TIERS, include_lowest=True),
        index=y.index,
        name="PriceTier",
    )
