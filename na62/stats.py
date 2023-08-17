from typing import Tuple, Union, Callable

import lmfit
import numpy as np
import pandas as pd
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt


def gaussian_wrapper(h: tuple[np.ndarray, np.ndarray], bins_center: np.ndarray) -> lmfit.model.ModelResult:
    gmodel = GaussianModel()
    pars = gmodel.guess(h, x=bins_center)
    return gmodel.fit(h, pars, x=bins_center, weights=1/(np.sqrt(h)))


def perform_fit(data: pd.DataFrame, *, bins: int,
                display_range: Tuple[int, int], fit_range: Union[Tuple[int, int], None] = None,
                ax: Union[None, plt.Axes] = None, plot: bool = False,
                model_wrapper: Union[Callable, None] = None) -> lmfit.model.ModelResult:

    # Compute the correct range and binning
    if fit_range is None:
        fit_range = display_range
    drange_size = display_range[1]-display_range[0]
    frange_size = fit_range[1] - fit_range[0]
    fit_bins = bins
    if drange_size != frange_size:
        fit_bins = int(bins*frange_size/drange_size)

    # Histogramming and bin centers (assuming constant binning)
    h, bins = np.histogram(data, bins=fit_bins, range=fit_range)
    bin_size = bins[1]-bins[0]
    bins_center = np.array([_ + bin_size/2 for _ in bins[:-1]])

    # Discard empty bins
    non_zero = np.nonzero(h)
    h = h[non_zero]
    bins_center = bins_center[non_zero]

    # Performing the fit
    if model_wrapper is None:
        model_wrapper = gaussian_wrapper
    out = model_wrapper(h, bins_center)

    # Drawing if needed
    if ax is not None:
        plot = True

    if plot:
        if ax is None:
            ax = plt.figure().gca()
        data.hist(bins=100, range=display_range, ax=ax)
        ax.plot(bins_center, out.best_fit, "-", label="Fit")
        ax.legend()
    return out
