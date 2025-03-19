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

from eztaox.kernels import direct, quasisep


class MultiVarModel(eqx.Module):
    """An interface for modeling multivariate/mutli-band time series using
    Gaussian Process (GP).

    This interface only takes GP kernels that can be evaluated using the
    scalable method of `DFM+17 <https://arxiv.org/abs/1703.09710>`. This
    interface allows fitting for the mean of the time series, additional
    variance to the uncertainty, and time delays between each uni-variate/
    single-band time series.

    Args:
        X (JAXArray): Input data containing time and band indices as a tuple.
        y (JAXArray | NDArray): Observed data values.
        yerr (JAXArray | NDArray): Observational errors.
        kernel (quasisep.Quasisep): A GP kernel from kernels.quasisep.
        **kwargs: Additional keyword arguments.
            zero_mean (bool): If True, assumes zero-mean GP. Defaults to True.
            has_jitter (bool): If True, assumes the input observational erros
                are underestimated. Defaults to False.
            has_lag (bool): If True, assumes time delays between time series in
                each band. Defaults to False.

    Raises:
        TypeError: If kernel is not one from kernels.quasisep.
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
        kernel: quasisep.Quasisep,
        **kwargs,
    ) -> None:
        if not isinstance(kernel, quasisep.Quasisep):
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
        kernel = quasisep.MultibandLowRank(
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
        """Calculate the log probability of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log probability of the input parameters.
        """
        gp, inds = self._build_gp(params)
        return gp.log_probability(y=self.y[inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp, inds = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[JAXArray, JAXArray]:
        """Make conditional GP prediction.

        Args:
            params (dict[str, JAXArray]): A dictionary containing model
                parameters.
            X (JAXArray): The time and band information for creating the
                conditional GP prediction.

        Returns:
            tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and
        """
        # transform time axis
        new_X, inds = self.lag_transform(X, self.has_lag, params)

        # build gp, cond
        gp, inds = self._build_gp(params)
        _, cond = gp.condition(self.y[inds], new_X)

        return cond.loc, jnp.sqrt(cond.variance)


class MultiVarModelFFT(MultiVarModel):
    """MultiVarModelFFT is a subclass of MultiVarModel for modeling multivariate
    time series data using Gaussian Processes with FFT-based transfer functions.

    This class extends the MultiVarModel by adding support for full-rank
    cross-band covariance matrices and user-defined transfer functions.

    The transfer_function needs to have the form:
    def f(X, **kwargs):
        # Some calculation
        p = jax.scipy.stats.norm.pdf(X[0], 5)
        return p
    See transfer_functions.py module

    Args:
        has_decorrelation (bool): Whether to add a decorrelation matrix to the
            kernel. Default is False.
        transfer_function (None | Callable): User-defined transfer function to
            use. Default is None.
    """

    has_decorrelation: bool = False
    transfer_function: None | Callable = None

    def __init__(
        self,
        X: JAXArray,
        y: JAXArray | NDArray,
        yerr: JAXArray | NDArray,
        kernel: tinygp.kernels.Kernel,
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
            kernel = direct.MultibandLowRank(
                amplitudes=jnp.exp(log_amps),
                kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
            )
        # full transfer function calculation
        else:
            kernel = direct.MultibandFFT(
                amplitudes=jnp.exp(log_amps),
                kernel=self.kernel_def(jnp.exp(params["log_kernel_param"])),
                transfer_function=jax.tree_util.Partial(self.transfer_function),
                **params,
            )
        # add the decorrelation matrix
        if self.has_decorrelation is True:
            nBand = params["log_amp_delta"].size + 1
            log_diagonal = jnp.zeros(nBand)
            kernel = direct.MultibandFullRank(
                kernel, jnp.exp(log_diagonal), params["off_diagonal"]
            )

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
    """An interface for modeling univariate/single-band time series using
    Gaussian Process (GP).

    This interface only takes GP kernels that can be evaluated using the
    scalable method of `DFM+17 <https://arxiv.org/abs/1703.09710>`. This
    interface allows fitting for the mean of the time series and additional
    variance to the uncertainty.

    Args:
        t (JAXArray): The time points of the observed data.
        y (JAXArray): The observed data.
        yerr (JAXArray): Observational errors.
        kernel (quasisep.Quasisep): A GP kernel from kernels.quasisep.
        **kwargs: Additional keyword arguments.
            zero_mean (bool): If True, assumes zero-mean GP. Defaults to True.
            has_jitter (bool): If True, assumes the input observational erros
                are underestimated. Defaults to False.

    Raises:
        TypeError: If kernel is not one from kernels.quasisep.
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
        kernel: quasisep.Quasisep,
        **kwargs,
    ) -> None:
        if not isinstance(kernel, quasisep.Quasisep):
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
        """Calculate the log probability of the input parameters.

        Args:
            params (dict[str, JAXArray]): Model parameters.

        Returns:
            JAXArray: Log probability of the input parameters.
        """
        gp = self._build_gp(params)
        return gp.log_probability(self.y[self.inds])

    def sample(self, params: dict[str, JAXArray]) -> None:
        """A convience function for intergrating with numpyro for MCMC sampling.

        Args:
            params (dict[str, JAXArray]): Model parameters.
        """
        gp = self._build_gp(params)
        numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[self.inds])

    @eqx.filter_jit
    def pred(
        self, params: dict[str, JAXArray], t: JAXArray | NDArray
    ) -> tuple[JAXArray, JAXArray]:
        """Make conditional GP prediction.

        Args:
            params (dict[str, JAXArray]): A dictionary containing model
                parameters.
            X (JAXArray): The time and band information for creating the
                conditional GP prediction.

        Returns:
            tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and
        """
        _, cond = self._build_gp(params).condition(self.y[self.inds], t)
        return cond.loc, jnp.sqrt(cond.variance)
