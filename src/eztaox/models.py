"""Light curve models module."""
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from jax.tree_util import PyTreeDef
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import mb_kernel


class MultiVarModel(eqx.Module):
    X: JAXArray
    y: JAXArray = eqx.field(converter=jnp.asarray)
    diag: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: PyTreeDef
    zero_mean: bool = True
    has_jitter: bool = False
    has_lag: bool = False

    def __init__(self, X, y, yerr, kernel, **kwargs) -> None:
        self.X = X
        self.diag = yerr**2
        self.y = y
        self.kernel_def = jax.tree_util.tree_flatten(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)
        self.has_lag = kwargs.get("has_lag", False)

    def lag_transform(
        self, has_lag, nBand, params
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        if has_lag is True:
            lags = jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)
        else:
            lags = jnp.zeros(nBand)
        t, band = self.X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def amp_transform(self, params) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_delta"]), 0, 0.0)

    @staticmethod
    def mean_func(zero_mean, nBand, params, X) -> JAXArray:
        if zero_mean is True:
            means = jnp.zeros(nBand)
        else:
            means = params["mean"]
        return means[X[1]]

    def log_prob(self, params) -> JAXArray:
        zero_mean = self.zero_mean
        has_lag = self.has_lag
        has_jitter = self.has_jitter
        return self.__call__(zero_mean, has_jitter, has_lag, params)

    def __call__(self, zero_mean, has_jitter, has_lag, params) -> JAXArray:
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(MultiVarModel.mean_func, zero_mean, log_amps.shape[0], params)

        # time axis transform: t and band are not sorted,
        # inds gives the sorted indices for the new_t
        X, inds = self.lag_transform(has_lag, log_amps.shape[0], params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        kernel = mb_kernel(
            amplitudes=jnp.exp(log_amps),
            kernel=jax.tree.unflatten(
                self.kernel_def, jnp.exp(params["log_kernel_param"])
            ),
        )

        gp = GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diags,
            mean=means,
            assume_sorted=True,
        )

        return -gp.log_probability(self.y[inds])

    def sample(self, params):
        zero_mean = self.zero_mean
        has_lag = self.has_lag
        has_jitter = self.has_jitter
        return self._sample(zero_mean, has_jitter, has_lag, params)

    def _sample(self, zero_mean, has_jitter, has_lag, params):
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(MultiVarModel.mean_func, zero_mean, log_amps.shape[0], params)

        # time axis transform
        X, inds = self.lag_transform(has_lag, log_amps.shape[0], params)
        t = X[0]
        band = X[1]

        # add jitter to the diagonal
        if has_jitter is True:
            diags = self.diag[inds] + (jnp.exp(params["log_jitter"]) ** 2)[band[inds]]
        else:
            diags = self.diag[inds]

        # def kernel
        kernel = mb_kernel(
            amplitudes=jnp.exp(log_amps),
            kernel=jax.tree.unflatten(
                self.kernel_def, jnp.exp(params["log_kernel_param"])
            ),
        )

        gp = GaussianProcess(
            kernel,
            (t[inds], band[inds]),
            diag=diags,
            mean=means,
            assume_sorted=True,
        )

        return numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])


class UniVarModel(eqx.Module):
    t: JAXArray = eqx.field(converter=jnp.asarray)
    y: JAXArray = eqx.field(converter=jnp.asarray)
    yerr: JAXArray = eqx.field(converter=jnp.asarray)
    inds: JAXArray = eqx.field(converter=jnp.asarray)
    kernel_def: PyTreeDef
    zero_mean: bool = True
    has_jitter: bool = False

    def __init__(self, t, y, yerr, kernel, **kwargs) -> None:
        self.t = t
        self.y = y
        self.yerr = yerr
        self.inds = jnp.argsort(t)
        self.kernel_def = jax.tree_util.tree_flatten(kernel)[1]
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)

    @staticmethod
    def mean_func(zero_mean, params, X) -> JAXArray:
        if zero_mean is True:
            mean = jnp.zeros(())
        else:
            mean = params["mean"]
        return mean

    def log_prob(self, params) -> JAXArray:
        zero_mean = self.zero_mean
        has_jitter = self.has_jitter
        return self.__call__(zero_mean, has_jitter, params)

    def __call__(self, zero_mean, has_jitter, params) -> JAXArray:
        # log mean
        mean = partial(UniVarModel.mean_func, zero_mean, params)

        # add jitter to the diagonal
        if has_jitter is True:
            diags = self.yerr**2 + jnp.exp(params["log_jitter"]) ** 2
        else:
            diags = self.yerr**2

        # def kernel
        kernel = jax.tree.unflatten(
            self.kernel_def, jnp.exp(params["log_kernel_param"])
        )

        gp = GaussianProcess(
            kernel,
            self.t[self.inds],
            diag=diags[self.inds],
            mean=mean,
            assume_sorted=True,
        )

        return -gp.log_probability(self.y[self.inds])

    def sample(self, params):
        zero_mean = self.zero_mean
        has_jitter = self.has_jitter
        return self._sample(zero_mean, has_jitter, params)

    def _sample(self, zero_mean, has_jitter, params):
        # log mean
        means = partial(UniVarModel.mean_func, zero_mean, params)

        # add jitter to the diagonal
        if has_jitter is True:
            diags = self.yerr**2 + jnp.exp(params["log_jitter"]) ** 2
        else:
            diags = self.yerr**2

        # def kernel
        kernel = jax.tree.unflatten(
            self.kernel_def, jnp.exp(params["log_kernel_param"])
        )

        gp = GaussianProcess(
            kernel,
            self.t[self.inds],
            diag=diags[self.inds],
            mean=means,
            assume_sorted=True,
        )

        return numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[self.inds])
