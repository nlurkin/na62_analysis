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
              label: str = "Data",
              ax: Union[None, plt.Axes] = None) -> int:
    h, bins = np.histogram(df, bins=bins, range=range)
    if errors == "normal":
        errors = np.sqrt(h)
    else:
        errors = None

    if ax is None:
        ax = plt
    ax.errorbar(get_bin_center(bins), h, fmt="k,",
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


def stack_mc_scale(dfs: List[pd.Series], *,
             bins: Union[int, None] = None, range: Union[int, None] = None,
             labels: Union[None, List[str]] = None,
             weights: Union[int, List[int]] = 1,
             ndata: Union[None, int] = None,
             ax: Union[None, plt.Axes] = None) -> Dict[str, int]:

    if isinstance(weights, int):
        weights = [weights]*len(dfs)
    if not labels:
        labels = [None]*len(dfs)

    hlist = []
    for df, weight, label in zip(dfs, weights, labels):
        hlist.append((df, np.ones(shape=df.shape)*weight, label))

    hlist = sorted(hlist, key=lambda x: sum(x[1]))
    sum_mc = sum([sum(_[1]) for _ in hlist])

    if ax is None:
        ax = plt

    ax.hist([_[0] for _ in hlist], weights=[_[1]*ndata/sum_mc for _ in hlist], bins=bins,
             range=range, stacked=True, label=[_[2] for _ in hlist])

    return {_[2]: sum(_[1]*ndata/sum_mc) for _ in hlist}


def stack_mc_flux(dfs: Dict[str, pd.Series], normalizations: Dict[str, int], *,
                  bins: Union[int, None] = None, range: Union[int, None] = None,
                  labels: Union[None, List[str]] = None,
                  kaon_flux: Union[None, int] = None,
                  ax: Union[None, plt.Axes] = None) -> Dict[str, int]:

    if not labels:
        labels = [None]*len(dfs)

    hlist = []
    for sample, label in zip(dfs, labels):
        if label is None:
            label = sample
        hlist.append((dfs[sample], np.ones(shape=dfs[sample].shape) *
                     constants.kaon_br_map[sample]/normalizations[sample], label))

    hlist = sorted(hlist, key=lambda x: sum(x[1]))

    if ax is None:
        ax = plt

    ax.hist([_[0] for _ in hlist], weights=[_[1]*kaon_flux for _ in hlist], bins=bins,
             range=range, stacked=True, label=[_[2] for _ in hlist])

    return {_[2]: sum(_[1]*kaon_flux) for _ in hlist}


def plot_variations(x: List, y: List, nominal_cut: int, name: str, cut_name: str, ax: Union[None, plt.Axes]) -> None:
    """
    Display a standard plot for systematic check of cut variation:
      - Plot the value corresponding to the nominal cut in a specific colour
      - Plot all other values, including uncorrelated uncertainties (as sqrt(abs(err**2-err_nominal**2)))
      - Plot a band corresponding to the statistical uncertainty of the nominal value

    :param x: List of cut values
    :param y: List of result values
    :param nominal_cut: Index of the nominal cut in the `x` and `y` lists
    :param name: Name of the result value (for display in the title and y axis)
    :param cut_name: Name of the variable on which the cut is done (displayed on the x axis)
    :param ax: Axes on which to draw the plots if specified. If None (default) create a new figure.
    """
    # Set aside the value for the nominal cut
    nominal_value = y[nominal_cut]
    nominal_std = nominal_value.s
    other_x = x[:nominal_cut] + x[nominal_cut+1:]
    other_y = y[:nominal_cut] + y[nominal_cut+1:]

    # For the x error bar (cosmetic)
    xerr = (x[1]-x[0])/2

    if not ax:
        ax = plt

    # Plot the values for the alternative cuts, using only uncorrelated errors
    ax.errorbar(other_x, [_.n for _ in other_y], yerr=[np.sqrt(np.abs(_.s**2-nominal_std**2)) for _ in other_y], capsize=2, ls="None", fmt=".", xerr=xerr, label="Varied cut (w/ uncorrelated errors)")
    # Plot the nominal value (with a different color)
    ax.errorbar(x[nominal_cut], nominal_value.n, yerr=0, capsize=2, ls="None", fmt=".", xerr=xerr, label="Nominal cut")
    ax.fill_between([x[0]-xerr, x[-1]+xerr], nominal_value.n-nominal_std/2, nominal_value.n + nominal_std/2, alpha=0.2, label="Statistical uncertainties")
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_title(f"{name} variation")
    ax.set_xlabel(f"{cut_name} cut")
    ax.set_ylabel(name)
    ax.legend()