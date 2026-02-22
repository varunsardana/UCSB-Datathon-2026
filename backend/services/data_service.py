"""
Data service — helpers for reading processed CSV data.

Used by analytics and disaster routers to avoid repeating file I/O logic.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_disasters() -> pd.DataFrame:
    path = DATA_DIR / "disasters.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_analytics() -> pd.DataFrame:
    path = DATA_DIR / "regional_analytics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
