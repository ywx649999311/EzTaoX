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


class LCModel(eqx.Module):
    X: JAXArray
    y: JAXArray
    diag: JAXArray
    kernel_def: PyTreeDef
    has_lag: bool = False
    zero_mean: bool = True
    has_jitter: bool = False

    def __init__(self, X, y, diag, kernel, **kwargs) -> None:
        self.X = X
        self.diag = diag
        self.y = y
        self.kernel_def = jax.tree_util.tree_flatten(kernel)[1]
        self.has_lag = kwargs.get("has_lag", False)
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_jitter = kwargs.get("has_jitter", False)

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
        has_lag = self.has_lag
        zero_mean = self.zero_mean
        return self.__call__(has_lag, zero_mean, params)

    def __call__(self, has_lag, zero_mean, params) -> JAXArray:
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(LCModel.mean_func, zero_mean, log_amps.shape[0], params)

        # time axis transform
        X, inds = self.lag_transform(has_lag, log_amps.shape[0], params)
        t = X[0]
        band = X[1]

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
            diag=self.diag[inds],
            mean=means,
            assume_sorted=True,
        )

        return -gp.log_probability(self.y[inds])

    def sample(self, params):
        has_lag = self.has_lag
        zero_mean = self.zero_mean
        return self._sample(has_lag, zero_mean, params)

    def _sample(self, has_lag, zero_mean, params):
        # log amp + mean
        log_amps = self.amp_transform(params)
        means = partial(LCModel.mean_func, zero_mean, log_amps.shape[0], params)

        # time axis transform
        X, inds = self.lag_transform(has_lag, log_amps.shape[0], params)
        t = X[0]
        band = X[1]

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
            diag=self.diag[inds],
            mean=means,
            assume_sorted=True,
        )

        return numpyro.sample("gp", gp.numpyro_dist(), obs=self.y[inds])
