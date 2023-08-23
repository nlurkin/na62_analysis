from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import constants


def get_bin_center(bins: np.array) -> np.array:
    return bins[:-1] + (bins[1]-bins[0])/2


def hist_data(df: pd.Series, *,
              bins: Union[int, None] = None, range: Union[int, None] = None,
              errors: Union[str, None] = "normal",
              label: str = "Data") -> int:
    h, bins = np.histogram(df, bins=bins, range=range)
    if errors == "normal":
        errors = np.sqrt(h)
    else:
        errors = None
    plt.errorbar(get_bin_center(bins), h, fmt="k,",
                 yerr=errors, capsize=2, label=label)
    return len(df)


def compute_samples_weights(normalizations_dict: Dict[str, float]):
    normalized_mc = []
    # Normalize each sample relative to its original size and BR
    for sample in normalizations_dict:
        normalization = normalizations_dict[sample]
        br = constants.kaon_br_map[sample]
        normalized_mc.append(br/normalization)

    return np.array(normalized_mc)


def stack_mc(dfs: List[pd.Series], *,
             bins: Union[int, None] = None, range: Union[int, None] = None,
             labels: Union[None, List[str]] = None,
             weights: Union[int, List[int]] = 1,
             ndata: Union[None, int] = None
             ) -> Dict[str, int]:

    if isinstance(weights, int):
        weights = [weights]*len(dfs)

    hlist = []
    for df, weight, label in zip(dfs, weights, labels):
        hlist.append((df, np.ones(shape=df.shape)*weight, label))

    hlist = sorted(hlist, key=lambda x: sum(x[1]))
    sum_mc = sum([sum(_[1]) for _ in hlist])

    plt.hist([_[0] for _ in hlist], weights=[_[1]*ndata/sum_mc for _ in hlist], bins=bins,
             range=range, stacked=True, label=[_[2] for _ in hlist])

    return {_[2]: sum(_[1]) for _ in hlist}