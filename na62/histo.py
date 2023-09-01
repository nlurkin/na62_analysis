from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import constants

disable_plotting = False

class Histogram:
    def __init__(self, histogram, bins, entries):
        self.histogram = histogram
        self.bins = bins
        self.entries = entries

    def merge(self, other):
        self.histogram =+ other.histogram
        self.entries += other.entries

        return self

def get_bin_center(bins: np.array) -> np.array:
    return bins[:-1] + (bins[1]-bins[0])/2


def compute_samples_weights(normalizations_dict: Dict[str, float]):
    normalized_mc = []
    # Normalize each sample relative to its original size and BR
    for sample in normalizations_dict:
        normalization = normalizations_dict[sample]
        br = constants.kaon_br_map[sample]
        normalized_mc.append(br/normalization)

    return np.array(normalized_mc)


################################################################
# Histograming of Data
################################################################

def hist_data(df: pd.Series, *,
              bins: Union[int, None] = None, range: Union[int, None] = None,
              errors: Union[str, None] = "normal",
              label: str = "Data",
              ax: Union[None, plt.Axes] = None) -> Union[int, Histogram]:

    h = prepare_hist_data(df, bins=bins, range=range)

    if disable_plotting:
        return h

    return _hist_data(h, bins=bins, range=range, errors=errors, label=label, ax=ax)


def prepare_hist_data(df: pd.Series, *,
              bins: Union[int, None] = None, range: Union[int, None] = None) -> int:
    h, bins = np.histogram(df, bins=bins, range=range)
    h = Histogram(h, bins, len(df))

    return h


def _hist_data(histogram: Histogram, *,
              bins: Union[int, None] = None, range: Union[int, None] = None,
              errors: Union[str, None] = "normal",
              label: str = "Data",
              ax: Union[None, plt.Axes] = None) -> int:
    if errors == "normal":
        errors = np.sqrt(histogram.histogram)
    else:
        errors = None

    if ax is None:
        ax = plt
    ax.errorbar(get_bin_center(histogram.bins), histogram.histogram, fmt="k,",
                 yerr=errors, capsize=2, label=label)
    return histogram.entries


################################################################
# Stacking of MC with Scaling to data
################################################################

def stack_mc_scale(dfs: List[pd.Series], *,
             bins: Union[int, None] = None, range: Union[int, None] = None,
             labels: Union[None, List[str]] = None,
             weights: Union[int, List[int]] = 1,
             ndata: int = 1,
             ax: Union[None, plt.Axes] = None) -> Union[Dict[str, int], List[Histogram]]:

    if not isinstance(dfs, List):
        dfs = [dfs]
    if isinstance(weights, int):
        weights = [weights]*len(dfs)

    hlist = []
    for df, weight in zip(dfs, weights):
        h = prepare_for_stack_scale(df, bins=bins, range=range, weight=weight)
        hlist.append(h)

    if disable_plotting:
        return hlist

    return _stack_mc_scale(hlist, labels=labels, ndata=ndata, ax=ax)


def prepare_for_stack_scale(df: pd.Series, *,
             bins: Union[int, None] = None, range: Union[int, None] = None,
             weight: int = 1) -> int:

    weights = np.ones(shape=df.shape)*weight

    h, bins = np.histogram(df, weights=weights, bins=bins, range=range)
    h = Histogram(h, bins, sum(weights))

    return h


def _stack_mc_scale(hlist: List[Histogram], *,
             labels: Union[None, List[str]] = None,
             ndata: int = 1,
             ax: Union[None, plt.Axes] = None) -> Dict[str, int]:

    hlist = sorted(hlist, key=lambda x: x.entries)
    sum_mc = sum([_.entries for _ in hlist])
    sum_mc = 1 if sum_mc==0 else sum_mc

    if ax is None:
        ax = plt

    width = hlist[0].bins[1] - hlist[0].bins[0]
    prev_bottom = 0
    for h, label in zip(hlist, labels):
        heights = h.histogram * ndata/sum_mc
        ax.bar(get_bin_center(h.bins), heights, width=width, bottom=prev_bottom, label=label)
        prev_bottom += heights

    return {label: _.entries*ndata/sum_mc for _,label in zip(hlist, labels)}


################################################################
# Stacking of MC with Scaling to flux
################################################################

def stack_mc_flux(dfs: Dict[str, pd.Series], normalizations: Dict[str, int], *,
                  bins: Union[int, None] = None, range: Union[int, None] = None,
                  labels: Union[None, List[str]] = None,
                  kaon_flux: Union[None, int] = None,
                  ax: Union[None, plt.Axes] = None) -> Dict[str, int]:

    if not isinstance(dfs, dict):
        dfs = {"dummy": dfs}
    if not labels:
        labels = [None]*len(dfs)

    hlist = []
    upd_labels = []
    for sample, label in zip(dfs, labels):
        if label is None:
            label = sample
        upd_labels.append(label)
        h = prepare_for_stack_flux(dfs[sample], bins=bins, range=range)
        hlist.append(h)

    if disable_plotting:
        return hlist

    return _stack_mc_flux(hlist, normalizations, labels=upd_labels, kaon_flux=kaon_flux, ax=ax)


def prepare_for_stack_flux(df: pd.Series, *,
                  bins: Union[int, None] = None, range: Union[int, None] = None) -> Dict[str, int]:

    weights = np.ones(shape=df.shape)
    h, bins = np.histogram(df, weights=weights, bins=bins, range=range)
    h = Histogram(h, bins, sum(weights))
    return h


def _stack_mc_flux(hlist: List[Histogram], normalizations: Dict[str, int], *,
                  labels: Union[None, List[str]] = None,
                  kaon_flux: Union[None, int] = None,
                  ax: Union[None, plt.Axes] = None) -> Dict[str, int]:

    if ax is None:
        ax = plt

    width = hlist[0].bins[1] - hlist[0].bins[0]
    prev_bottom = 0
    for h, label in zip(hlist, labels):
        heights = h.histogram * kaon_flux * constants.kaon_br_map[label] / normalizations[label]
        ax.bar(get_bin_center(h.bins), heights, width=width, bottom=prev_bottom, label=label)
        prev_bottom += heights

    return {label: _.entries*kaon_flux for _,label in zip(hlist, labels)}


################################################################
# Other
################################################################

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