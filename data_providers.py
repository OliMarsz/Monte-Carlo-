import pandas as pd
import numpy as np
import yfinance as yf

"""
Utility functions for converting between paths and dataframes, used in MCRunner and elsewhere.
Will likely add more functions here in the future as needed, 
such as for converting between different API formats, etc.
"""

def path_to_df(path, start="2020-01-01", freq="B"):
    path = np.asarray(path)
    idx = pd.date_range(start=start, periods=len(path), freq=freq)
    return pd.DataFrame({"Close": path}, index=idx)

