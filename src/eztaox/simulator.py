"""Light curve models module."""
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
from numpy.typing import NDArray
from tinygp import GaussianProcess
from tinygp.helpers import JAXArray

from eztaox.kernels import quasisep
from eztaox.ts_utils import _get_nearest_idx


class MultiVarSim(eqx.Module):
    base_kernel_def: Callable
    multiband_kernel: quasisep.Wrapper
    X: tuple[JAXArray, JAXArray]
    init_params: dict[str, JAXArray]
    nBand: int
    mean_func: Callable | None
    amp_scale_func: Callable | None
    lag_func: Callable | None
    zero_mean: bool
    has_lag: bool

    def __init__(
        self,
        base_kernel: quasisep.Quasisep,
        min_dt: float,
        max_dt: float,
        nBand: int,
        init_params: dict[str, JAXArray],
        multiband_kernel: quasisep.Wrapper | None = quasisep.MultibandLowRank,
        mean_func: Callable | None = None,
        amp_scale_func: Callable | None = None,
        lag_func: Callable | None = None,
        **kwargs,
    ) -> None:
        # make sim X
        simN = int(max_dt / min_dt) + 1
        ts, bands = [], []
        for i in range(nBand):
            ts.append(jnp.linspace(0, max_dt, simN))
            bands.append(jnp.full_like(ts[i], i, dtype=int))
        t = jnp.concat(ts)
        band = jnp.concat(bands)

        # assign fixed values
        inds = jnp.argsort(t)
        self.X = (t[inds], band[inds])
        self.nBand = nBand
        self.init_params = init_params

        # assign callables/classes
        self.base_kernel_def = jax.flatten_util.ravel_pytree(base_kernel)[1]
        self.multiband_kernel = multiband_kernel
        self.mean_func = mean_func
        self.amp_scale_func = amp_scale_func
        self.lag_func = lag_func

        # assign other attributes
        self.zero_mean = kwargs.get("zero_mean", True)
        self.has_lag = kwargs.get("has_lag", False)

        # compile funcs
        self._build_gp(self.init_params)

    @eqx.filter_jit
    def full(
        self, key: jax.random.PRNGKey, params: dict[str, JAXArray] | None = None
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        """Build a full Gaussian Process with the given parameters."""

        params = params if params is not None else self.init_params
        gp, inds = self._build_gp(params)

        return (self.X[0][inds], self.X[1][inds]), gp.sample(key)

    @eqx.filter_jit
    def random(
        self,
        nRand: int,
        lc_key: jax.random.PRNGKey,
        random_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
        """Sample a random Gaussian Process with the given parameters."""

        # get full light curve
        params = params if params is not None else self.init_params
        full_X, full_y = self.full(lc_key, params)

        # select randomly & return
        rand_inds = jnp.sort(jax.random.permutation(random_key, full_y.size)[:nRand])
        return (
            (full_X[0][rand_inds], full_X[1][rand_inds]),
            full_y[rand_inds],
        )

    # @eqx.filter_jit
    def fixed_input(
        self,
        sim_X: tuple[JAXArray | NDArray, JAXArray | NDArray],
        lc_key: jax.random.PRNGKey,
        params: dict[str, JAXArray] | None = None,
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
        """Sample a random Gaussian Process with the given parameters at fixed
        input time and labels."""
        # convert sim_X to JAXArray and ensure band is int
        sim_X = (jnp.asarray(sim_X[0]), jnp.asarray(sim_X[1]).astype(int))

        # get full light curve
        params = params if params is not None else self.init_params
        full_X, full_y = self.full(lc_key, params)

        # get indices for the input sim_X
        ts, bands, ys = [], [], []
        for i in range(self.nBand):
            full_band_mask = full_X[1] == i
            input_band_mask = sim_X[1] == i
            inds = jax.vmap(_get_nearest_idx, in_axes=(None, 0))(
                full_X[0][full_band_mask], sim_X[0][input_band_mask]
            )
            ts.append(full_X[0][full_band_mask][inds])
            bands.append(jnp.full_like(ts[i], i, dtype=int))
            ys.append(full_y[full_band_mask][inds])

        return ((jnp.concat(ts), jnp.concat(bands)), jnp.concat(ys))

    @eqx.filter_jit
    def _build_gp(
        self,
        params: dict[str, JAXArray],
    ) -> tuple[GaussianProcess, JAXArray]:
        # log amp + mean
        log_amp_scales = self.get_amp_scale(params)
        means = partial(self.get_mean, self.zero_mean, params)

        # time axis transform: new_inds gives the sorted indices for t, band,
        # after the lag transform
        if self.has_lag is False:
            inds = jnp.arange(self.X[0].size)
            new_t = self.X[0]
        else:
            new_X, inds = self.lag_transform(self.has_lag, params, self.X)
            new_t = new_X[0]

        # def kernel
        new_params = params.copy()
        new_params["amplitudes"] = jnp.exp(log_amp_scales)
        kernel = self.multiband_kernel(
            params=new_params,
            kernel=self.base_kernel_def(jnp.exp(new_params["log_kernel_param"])),
        )

        return (
            GaussianProcess(
                kernel,
                (new_t[inds], self.X[1][inds]),
                mean=means,
                assume_sorted=True,
            ),
            inds,
        )

    def get_mean(
        self, zero_mean: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> JAXArray:
        """Mean func for the Gaussian Process."""
        if zero_mean is True:
            mean = 0.0
        elif self.mean_func is not None:
            mean = self.mean_func(params, X)
        else:
            mean = self._default_mean_func(params, X)
        return mean

    def get_amp_scale(self, params: dict[str, JAXArray]) -> JAXArray:
        """Amplitude transform for the Gaussian Process."""
        if self.amp_scale_func is not None:
            return self.amp_scale_func(params)
        return self._default_amp_scale_func(params)

    def lag_transform(
        self, has_lag: bool, params: dict[str, JAXArray], X: JAXArray
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        """Lag transform for the Gaussian Process."""
        if has_lag is False:
            lags = jnp.zeros(self.nBand)
        elif self.lag_func is not None:
            lags = self.lag_func(params)
        else:
            lags = self._default_lag_func(params)

        t, band = X
        new_t = t - lags[band]
        inds = jnp.argsort(new_t)
        return (new_t, band), inds

    def _default_mean_func(self, params: dict[str, JAXArray], X: JAXArray) -> JAXArray:
        return jnp.atleast_1d(params["mean"])[X[1]]

    def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
        return jnp.insert(jnp.atleast_1d(params["log_amp_scale"]), 0, 0.0)

    def _default_lag_func(
        self, params: dict[str, JAXArray]
    ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
        return jnp.insert(jnp.atleast_1d(params["lag"]), 0, 0.0)


# class UniVarModel(MultiVarModel):
#     """A subclass of MultiVarModel for modeling univariate time series data
#     using Gaussian Processes with FFT-based transfer functions.

#     This class extends the MultiVarModel by adding support for full-rank
#     cross-band covariance matrices and user-defined transfer functions.

#     Args:
#         has_decorrelation (bool): Whether to add a decorrelation matrix to the
#             kernel. Default is False.
#         transfer_function (None | Callable): User-defined transfer function to
#             use. Default is None.
#     """

#     def __init__(
#         self,
#         t: JAXArray | NDArray,
#         y: JAXArray | NDArray,
#         yerr: JAXArray | NDArray,
#         kernel: tinygp.kernels.Kernel,
#         mean_func: Callable | None = None,
#         amp_scale_func: Callable | None = None,
#         **kwargs,
#     ) -> None:
#         """Initialize the UniVarModel2 with time, observed data, and kernel."""

#         inds = jnp.argsort(jnp.asarray(t))
#         X = (jnp.asarray(t)[inds], jnp.zeros_like(t, dtype=int))
#         y = jnp.asarray(y)[inds]
#         yerr = jnp.asarray(yerr)[inds]
#         base_kernel = kernel
#         nBand = 1
#         has_lag = False
#         super().__init__(
#             X,
#             y,
#             yerr,
#             base_kernel,
#             nBand,
#             mean_func=mean_func,
#             amp_scale_func=amp_scale_func,
#             has_lag=has_lag,
#             **kwargs,
#         )

#     def _default_amp_scale_func(self, params: dict[str, JAXArray]) -> JAXArray:
#         return jnp.array([0.0])

#     def lag_transform(
#         self, has_lag, params, X
#     ) -> tuple[tuple[JAXArray, JAXArray], JAXArray]:
#         return self.X, jnp.arange(self.X[0].size)

#     def pred(self, params, t) -> tuple[JAXArray, JAXArray]:
#         """Make conditional GP prediction.

#         Args:
#             params (dict[str, JAXArray]): A dictionary containing model
#                 parameters.
#             t (JAXArray): The time information for creating the
#                 conditional GP prediction.

#         Returns:
#             tuple[JAXArray, JAXArray]: A tuple of the mean GP prediction and
#         """
#         # build gp, cond
#         gp, inds = self._build_gp(params)
#         _, cond = gp.condition(self.y[inds], (t, jnp.zeros_like(t, dtype=int)))

#         return cond.loc, jnp.sqrt(cond.variance)
