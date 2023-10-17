from __future__ import annotations
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process.kernels import Kernel, Matern, RationalQuadratic, Exponentiation, DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from .. import Vector
from tqdm.auto import tqdm


"""
TODO:
- [ ] The GP uncertainty is unrealistically small.
- [ ] How to combine bootstrap CI with GP std?
- [ ] 2D GP and fit. 
    - Might require too much memory. Downsample or sparse GP.
    - Downsampling seems to be good.
"""

@dataclass
class FitResult:
    base: Vector
    mask: np.ndarray
    kernel: Kernel
    background: Vector
    bg_sigma: Vector
    peak: Vector
    residual: Vector
    popt: np.ndarray
    pcov: np.ndarray
    
    def plot_gp_fit(self, ax=None, mask=True):
        if ax is None:
            fig, ax = plt.subplots()
        if mask:
            low_i = np.argmax(self.mask)
            high_i = len(self.mask) - np.argmax(self.mask[::-1]) - 1
            low = self.base.X[low_i]
            high = self.base.X[high_i]
            ax.axvspan(low, high, alpha=0.1, color='k')

        self.base.plot(ax=ax, label='data')
        self.background.plot(ax=ax, label='background (GP) fit')
        bg = self.background.values
        sigma = self.bg_sigma.values
        ax.fill_between(self.bg_sigma.X, bg - 2*sigma, bg + 2*sigma, alpha=0.1, label='2$\sigma$')
        ax.set_title('Original Spectrum vs. Background Fit')
        return  ax
        
    def plot_peak_fit(self, ax=None, se: bool = True):
        if ax is None:
            fig, ax = plt.subplots()
        #self.residual.plot(ax=ax, label='residual')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)  # Zero line
        (self.base - self.background).plot(ax=ax, label='residual')
        self.peak.plot(ax=ax, label='peak (Gaussian) fit')
        if se:
            X = self.base.X
            standard_errors = np.sqrt(np.diag(self.pcov))
            peak_low = gaussian(X, *(self.popt - standard_errors))
            peak_high = gaussian(X, *(self.popt + standard_errors))
            ax.fill_between(X, peak_low, peak_high, alpha=0.1, label='1$\sigma$')
        ax.set_title('Residual vs. Gaussian Fit')
        ax.set_ylabel('Residual Counts')
        return ax

    def plot(self, ax=None, mask=True):
        ax = self.plot_gp_fit(ax=ax, mask=mask)
        total = self.peak + self.background
        total.plot(ax=ax, label='total fit')
        bg = self.base - self.peak
        bg.plot(ax=ax, label='adjusted bg fit')
        return ax

    def adjusted_bg(self):
        return self.base - self.peak


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / (2 * stddev))**2)


def fit(vectors: list[Vector] | Vector, mask: np.ndarray, kernel: Kernel = Matern(nu=0.5),
        gp: GaussianProcessRegressor | None = None) -> list[FitResult] | FitResult:
    match vectors:
        case Vector():
            return fit_vector(vectors, mask, kernel, gp=gp)
        case _:
            return fit_list(vectors, mask, kernel, gp=gp)


def fit_list(vectors: list[Vector], mask: np.ndarray, kernel: Kernel = Matern(nu=0.5), gp=None) -> list[FitResult]:
    
    results: list[FitResult] = []
    
    for vec in tqdm(vectors):
        result = fit_vector(vec, mask, kernel)
        results.append(result)
    
    return results


def fit_vector(vec: Vector, mask: np.ndarray, kernel: Kernel = Matern(nu=0.5), gp=None) -> FitResult:
    if gp is None:
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    X, y = vec.X, vec.values
    X_mask = X[~mask]
    y_mask = y[~mask]
    X_mask = X_mask[:, np.newaxis]
    X = X[:, np.newaxis]

    gp.fit(X_mask, y_mask)
    background_fit, sigma = gp.predict(X, return_std=True)
    bg = vec.clone(values=background_fit, name='background (GP) fit')
    bg_sigma = vec.clone(values=sigma, name='background sigma (GP) fit')
    
    # Calculate residuals
    residuals = vec.values - background_fit
    X = vec.X
    
    # Fit Gaussian to residuals
    mean = X[np.argmax(residuals)]
    # prevent numerical instability
    sigma[sigma < 1e-6] = 1e-6
    popt, pcov = curve_fit(gaussian, X, residuals, p0=[residuals.max(), mean, 50], sigma=1.0/sigma, absolute_sigma=True)
    standard_errors = np.sqrt(np.diag(pcov))
    amplitude_fitted, mean_fitted, stddev_fitted = popt
    gaussian_fit = gaussian(X, amplitude_fitted, mean_fitted, stddev_fitted)
    peak = vec.clone(values=gaussian_fit, name='peak (Gaussian) fit')
    
    # Combined fit and final residuals
    combined_fit = background_fit + gaussian_fit
    final_residual = vec - combined_fit
    
    # Append to results
    result = FitResult(base=vec, mask=mask, kernel=kernel, background=bg,
                       bg_sigma=bg_sigma, peak=peak, residual=final_residual, popt=popt, pcov=pcov)
    return result
