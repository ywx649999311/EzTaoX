"""Light curve models module."""
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpyro
import tinygp
from numpy.typing import NDArray
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import MultibandDeltaQuasi, MultibandDelta, MultibandFFT, MultibandDecorrelation


class MultiVarModel(eqx.Module):
    """
    MultiVarModel is a class for modeling multivariate time series data using Gaussian Processes.

    Attributes:
        X (JAXArray): The input features, consisting of time and band indices.
        y (JAXArray): The observed data, typically magnitudes.
        diag (JAXArray): The diagonal elements of the covariance matrix, typically representing the variance of the observations.
        kernel_def (Callable): The kernel function used in the Gaussian Process.
        zero_mean (bool): Whether to use a zero mean function. Default is True.
        has_jitter (bool): Whether to add jitter to the diagonal of the covariance matrix. Default is False.
        has_lag (bool): Whether to apply a lag transformation to the time axis. Default is False.

    Methods:
        __init__(self, X, y, yerr, kernel, **kwargs):
            Initializes the MultiVarModel with the given data, kernel, and optional parameters.

        lag_transform(self, X, has_lag, params):
            Applies a lag transformation to the time axis if has_lag is True.

        amp_transform(self, params):
            Transforms the amplitude parameters.

        mean_func(zero_mean, nBand, params, X):
            Computes the mean function for the Gaussian Process.

        _build_gp(self, params):
            Builds the Gaussian Process model with the given parameters.

        log_prob(self, params):
            Computes the log probability of the observed data under the Gaussian Process model.

        sample(self, params):
            Samples from the Gaussian Process model using the given parameters.

        pred(self, params, X):
            Makes predictions using the Gaussian Process model for the given input features.
    """
    X: JAXArray
    y: JAXArray = eqx.field(converter=jnp.asarray)
    diag: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: Callable
    zero_mean: bool = True
    has_jitter: bool = False
    has_lag: bool = False

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:

        if not isinstance(kernel, tinygp.kernels.quasisep.Quasisep):
            raise TypeError("This model only takes quasiseperable kernels.")

        self.X = (jnp.asarray(X[0]), jnp.asarray(X[1], dtype=int)) 
        self.diag = yerr**2
        self.y = y
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)

    def lag_transform(
        self, X: JAXArray, has_lag: bool, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        if has_lag is True:
            lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
        else:
            nBand = params["log_amp_delta"].size + 1
            lags = jnp.zeros(nBand)
        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def amp_transform(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)

    @staticmethod
    def mean_func(
        zero_mean: bool, nBand: int, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        if zero_mean is True:
            means = jnp.zeros(nBand)
        else:
            means = jnp.atleast_1d(params["mean"])
        return means[X[1]]

    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(
            MultiVarModel.mean_func, self.zero_mean, log_amps.shape[0], params
        )

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(self.X, self.has_lag, params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        kernel = MultibandDeltaQuasi(
            amplitudes=jnp.exp(log_amps),
            kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
        )

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags,
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp, inds = self._build_gp(params)
        return gp.log_probability(y=self.y[inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        # transform time axis
        new_X, inds = self.lag_transform(X, self.has_lag, params)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)


class MultiVarModelFFT(MultiVarModel):
    """
    MultiVarModelFFT is a subclass of MultiVarModel for modeling multivariate time series data using Gaussian Processes with FFT-based transfer functions.

    This class extends the MultiVarModel by adding support for decorrelation matrices and user-defined transfer functions.
    The transfer_function needs to have the form:
    def f(X, **kwargs):
        # Some calculation
        p = jax.scipy.stats.norm.pdf(X[0], 5)
        return p
    See transfer_functions.py module

    Attributes:
        has_decorrelation (bool): Whether to add a decorrelation matrix to the kernel. Default is False.
        transfer_function (None | Callable): User-defined transfer function to use. Default is None.

    Methods:
        __init__(self, X, y, yerr, kernel, **kwargs):
            Initializes the MultiVarModelFFT with the given data, kernel, and optional parameters.

        _build_gp(self, params):
            Builds the Gaussian Process model with the given parameters, including the transfer function and decorrelation matrix if specified.
    """
    has_decorrelation: bool = False
    transfer_function: None | Callable = None

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        self.X = (jnp.asarray(X[0]), jnp.asarray(X[1], dtype=int)) 
        self.diag = yerr**2
        self.y = y
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)
        self.has_decorrelation = kwargs.get("has_decorrelation", False)
        self.transfer_function = kwargs.get("transfer_function", None)

    def _build_gp(
        self, params: dict[str, JAXArray]
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(
            MultiVarModel.mean_func, self.zero_mean, log_amps.shape[0], params
        )

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(self.X, self.has_lag, params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        if self.transfer_function is None:
            kernel = MultibandDelta(
                amplitudes=jnp.exp(log_amps),
                kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
            )
        # full transfer function calculation
        else:
            kernel = MultibandFFT(
                amplitudes=jnp.exp(log_amps),
                kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
                transfer_function=jax.tree_util.Partial(self.transfer_function),
                **params
            )
        # add the decorrelation matrix
        if self.has_decorrelation is True:
            nBand = params["log_amp_delta"].size + 1
            log_diagonal = jnp.zeros(nBand)
            kernel = MultibandDecorrelation(kernel, jnp.exp(log_diagonal), params["off_diagonal"])

        return (
            GaussianProcess(
                kernel,
                (t[inds], band[inds]),
                diag=diags,
                mean=means,
            ),
            inds,
        )


class UniVarModel(eqx.Module):
    """
    UniVarModel is a class for modeling univariate time series data using Gaussian Processes.

    Attributes:
        t (JAXArray): The time points of the observed data.
        y (JAXArray): The observed data.
        yerr (JAXArray): The errors of the observed data.
        inds (JAXArray): The indices that sort the time points.
        kernel_def (Callable): The kernel function used in the Gaussian Process.
        zero_mean (bool): Whether to use a zero mean function. Default is True.
        has_jitter (bool): Whether to add jitter to the diagonal of the covariance matrix. Default is False.

    Methods:
        __init__(self, t, y, yerr, kernel, **kwargs):
            Initializes the UniVarModel with the given time series data, kernel, and optional parameters.

        mean_func(zero_mean, params, X):
            Computes the mean function for the Gaussian Process.

        _build_gp(self, params):
            Builds the Gaussian Process model with the given parameters.

        log_prob(self, params):
            Computes the log probability of the observed data under the Gaussian Process model.

        sample(self, params):
            Samples from the Gaussian Process model using the given parameters.

        pred(self, params, t):
            Makes predictions using the Gaussian Process model for the given time series data.
    """
    t: JAXArray = eqx.field(converter=jnp.asarray)
    y: JAXArray = eqx.field(converter=jnp.asarray)
    yerr: JAXArray = eqx.field(converter=jnp.asarray)
    inds: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: Callable
    zero_mean: bool = True
    has_jitter: bool = False

    def __init__(
        self,
        t: JAXArray | NDArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.quasisep.Quasisep,
        **kwargs,
    ) -> None:
        if not isinstance(kernel, tinygp.kernels.quasisep.Quasisep):
            raise TypeError("This model only takes quasiseperable kernels.")
        self.t = t
        self.y = y
        self.yerr = yerr
        self.inds = jnp.argsort(t)
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)

    @staticmethod
    def mean_func(zero_mean, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        if zero_mean is True:
            mean = jnp.zeros(())
        else:
            mean = params["mean"]
        return mean

    def _build_gp(self, params: dict[str, JAXArray]) -> GaussianProcess:
        mean = partial(UniVarModel.mean_func, self.zero_mean, params)

        # add jitter to the diagonal
        if self.has_jitter is True:
            diags = self.yerr**2 + jnp.exp(params["log_jitter"]) ** 2
        else:
            diags = self.yerr**2

        # re-create kernel
        kernel = self.kernel_def(jnp.exp(params["log_kernel_param"]))
        return GaussianProcess(
            kernel,
            self.t[self.inds],
            diag=diags[self.inds],
            mean=mean,
            assume_sorted=True,
        )

    @eqx.filter_jit
    def log_prob(self, params: dict[str, JAXArray]) -> JAXArray:
        gp = self._build_gp(params)
        return gp.log_probability(self.y[self.inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        gp = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[self.inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], t: JAXArray | NDArray
    ) -> tuple[JAXArray, JAXArray]:
        _, cond = self._build_gp(params).condition(self.y[self.inds], t)
        return cond.loc, jnp.sqrt(cond.variance)