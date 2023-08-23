from typing import Callable, Tuple, Union

import lmfit
import numpy as np
import pandas as pd
from lmfit.models import GaussianModel
from matplotlib import pyplot as plt


def model_wrapper(models) -> Callable:
    prepared_models = [m(prefix=f"m{i}_") for i, m in enumerate(models)]

    def model(h: tuple[np.ndarray, np.ndarray], bins_center: np.ndarray) -> lmfit.model.ModelResult:
        pars = []
        for m in prepared_models:
            this_pars = m.guess(h, x=bins_center)
            # Remove estimated contribution from histogram
            h = h - m.eval(x=bins_center, params=this_pars)
            pars.append(this_pars)

        return sum(prepared_models[1:], prepared_models[0]), sum(pars[1:], pars[0])

    return model


def gaussian_wrapper(h: tuple[np.ndarray, np.ndarray], bins_center: np.ndarray) -> lmfit.model.ModelResult:
    """Fit a single Gaussian

    :param h: Histogram to fit
    :param bins_center: Histogram bin centers
    :return: Fit result
    """
    gmodel = GaussianModel()
    pars = gmodel.guess(h, x=bins_center)
    return gmodel, pars


def gaussian2_wrapper(h: tuple[np.ndarray, np.ndarray], bins_center: np.ndarray) -> lmfit.model.ModelResult:
    """Fit a double Gaussian, assuming similar mean but larger sigma for the second Gaussian

    :param h: Histogram to fit
    :param bins_center: Histogram bin centers
    :return: Fit result
    """

    return model_wrapper([GaussianModel, GaussianModel])(h, bins_center)


def perform_fit(data: Union[pd.Series, np.ndarray], *, bins: int,
                display_range: Tuple[int, int], fit_range: Union[Tuple[int, int], None] = None,
                ax: Union[None, plt.Axes] = None, plot: bool = False, fit_label: str = "Fit",
                model_wrapper: Union[Callable, None] = None) -> lmfit.model.ModelResult:
    """
    Configure and perform a fit of the input data according to the chosen model. The input data are raw
    data that are first binned, then the fit is performed on the histogram. The histogram and fit result
    can optionally be plotted.

    :param data: Data to histogram and fit (array format, either numpy array or pandas series)
    :param bins: Number of bins for the histogram
    :param display_range: Complete range of the histogram
    :param fit_range: Restricted range for fitting. If None, uses the same values as the display_range. (default None)
    :param ax: Axes on which to draw the histogram. If specified, automatically set 'plot = True'.
        If plot is True and ax is not specified, a new figure is created. (default None)
    :param plot: Whether to draw the histogram and results.
    :param fit_label: Label for the fit in the legend, if plotted. (default 'Fit')
    :param model_wrapper: Wrapper function defining the model to use for the fitting. (default None)
    :return: Fit result as a ModelResult from lmfit
    """

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
    model, pars = model_wrapper(h, bins_center)
    out = model.fit(h, pars, x=bins_center, weights=1/(np.sqrt(h)))

    # Drawing if needed
    if ax is not None:
        plot = True

    if plot:
        if ax is None:
            ax = plt.figure().gca()
        data.hist(bins=100, range=display_range, ax=ax)
        ax.plot(bins_center, out.best_fit, "-", label=fit_label)
        ax.legend()
    return out
