from typing import List, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import constants


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
    plt.errorbar(get_bin_center(bins), h, fmt="k,",
                 yerr=errors, capsize=2, label=label)


def compute_samples_weights(normalizations_dict: Dict[str, float]):
    normalized_mc = []
    # Normalize each sample relative to its original size and BR
    for sample in normalizations_dict:
        normalization = normalizations_dict[sample]
        br = constants.kaon_br_map[sample]
        normalized_mc.append(br/normalization)

    # This is the total normalized MC
    total_mc = np.sum(normalized_mc)

    # Normalize each sample with the total MC
    return np.array(normalized_mc) / total_mc


def stack_mc(dfs: List[pd.Series], *,
             bins: Union[int, None] = None, range: Union[int, None] = None,
             labels: Union[None, List[str]] = None,
             weights: Union[int, List[int]] = 1,
             ndata: Union[None, int] = None
             ):

    if isinstance(weights, int):
        weights = [weights]*len(dfs)

    hlist = []
    hweights = []
    for df, weight in zip(dfs, weights):
        data_factor = ndata / len(df) if ndata else 1
        hweights.append(np.ones(shape=df.shape)*weight*data_factor)
        hlist.append(df)

    plt.hist(hlist, weights=hweights, bins=bins,
             range=range, stacked=True, label=labels)
