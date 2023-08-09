from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_bin_center(bins: np.array) -> np.array:
    return bins[:-1] + (bins[1]-bins[0])/2


def hist_data(df: pd.Series, *,
              bins: Union[int, None] = None, range: Union[int, None] = None,
              errors: Union[str, None] = "normal",
              label: str = "Data"):
    h, bins = np.histogram(df, bins=bins, range=range)
    if errors == "normal":
        errors = np.sqrt(h)
    else:
        errors = None
    plt.errorbar(get_bin_center(bins), h, fmt="r,",
                 yerr=errors, capsize=2, label=label)

