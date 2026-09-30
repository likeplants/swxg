import os
import pandas as pd


monthly = pd.read_pickle(os.path.dirname(__file__) + "/example_monthly.pkl")
daily = pd.read_pickle(os.path.dirname(__file__) + "/example_daily.pkl")

__all__ = ["monthly", "daily"]

