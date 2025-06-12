"""Second-Order statistic functions for kernels"""
from collections.abc import Callable

import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import tinygp
from jax._src import dtypes
from numpy.typing import NDArray
from tinygp.helpers import JAXArray

from eztaox.kernels.quasisep import carma_acvf, carma_roots


class gpStat2(eqx.Module):
    """Base class for second-order statistics of Gaussian processes."""

    kernel_def: Callable

    def __init__(self, kernel: Callable) -> None:
        self.kernel_def = jax.flatten_util.ravel_pytree(kernel)[1]

    def _build_kernel(
        self, params: JAXArray | NDArray
    ) -> tuple[tinygp.kernels.Kernel, JAXArray]:
        kernel = self.kernel_def(jnp.asarray(params))
        return kernel, kernel.evaluate_diag(0.0)

    def acf(self, ts: JAXArray | NDArray, params: JAXArray | NDArray) -> JAXArray:
        kernel, amp2 = self._build_kernel(params)
        acvf = jax.vmap(kernel.evaluate, in_axes=(None, 0))(0.0, jnp.asarray(ts))
        return acvf / amp2

    def sf(self, ts: JAXArray | NDArray, params: JAXArray | NDArray) -> JAXArray:
        kernel, amp2 = self._build_kernel(params)
        acf = jax.vmap(kernel.evaluate, in_axes=(None, 0))(0.0, jnp.asarray(ts)) / amp2
        return jnp.sqrt(2 * amp2 * (1 - acf))


@jax.jit
def carma_rms(alpha: JAXArray | NDArray, beta: JAXArray | NDArray) -> JAXArray:
    alpha = jnp.atleast_1d(alpha)
    beta = jnp.atleast_1d(beta)
    _arroots = carma_roots(jnp.append(alpha, 1.0))
    _acf = carma_acvf(_arroots, alpha, beta * 1.0)
    return jnp.sqrt(jnp.abs(jnp.sum(_acf)))


@jax.jit
def carma_psd(
    f: JAXArray | NDArray, arparams: JAXArray | NDArray, maparams: JAXArray | NDArray
) -> JAXArray:
    """
    Return a function that computes CARMA Power Spectral Density (PSD).

    Args:
        arparams (array(float)): AR coefficients.
        maparams (array(float)): MA coefficients

    Returns:
        A function that takes in frequencies and returns PSD at the given frequencies.
    """
    arparams = jnp.append(jnp.array(arparams), 1.0)
    maparams = jnp.array(maparams)

    complex_dtype = dtypes.to_complex_dtype(arparams.dtype)

    # init terms
    num_terms = jnp.zeros(1, dtype=complex_dtype)
    denom_terms = jnp.zeros(1, dtype=complex_dtype)

    for i, param in enumerate(maparams):
        num_terms += param * jnp.power(2 * jnp.pi * f * (1j), i)

    for k, param in enumerate(arparams):
        denom_terms += param * jnp.power(2 * jnp.pi * f * (1j), k)

    num = jnp.abs(jnp.power(num_terms, 2))
    denom = jnp.abs(jnp.power(denom_terms, 2))

    return num / denom


@jax.jit
def carma_acf(
    t: JAXArray | NDArray, arparams: JAXArray | NDArray, maparams: JAXArray | NDArray
) -> JAXArray:
    """
    Return a function that computes the model autocorrelation function (ACF) of CARMA.

    Args:
        arparams (array(float)): AR coefficients.
        maparams (array(float)): MA coefficients.

    Returns:
        A function that takes in time lags and returns ACF at the given lags.
    """

    roots = carma_roots(jnp.append(arparams, 1.0))
    autocorr = carma_acvf(roots, arparams, maparams)
    carma_amp = carma_rms(arparams, maparams)

    R = 0
    for i, r in enumerate(roots):
        R += autocorr[i] * jnp.exp(r * t)

    return jnp.real(R / carma_amp**2)


@jax.jit
def carma_sf(
    t: JAXArray | NDArray, arparams: JAXArray | NDArray, maparams: JAXArray | NDArray
) -> JAXArray:
    """
    Return a function that computes the CARMA structure function (SF).

    Args:
        arparams (array(float)): AR coefficients.
        maparams (array(float)): MA coefficients.

    Returns:
        A function that takes in time lags and returns CARMA SF at the given lags.
    """
    amp2 = carma_rms(arparams, maparams) ** 2
    return jnp.sqrt(2 * amp2 * (1 - carma_acf(t, arparams, maparams)))


@jax.jit
def drw_psd(
    f: JAXArray | NDArray, tau: JAXArray | float, amp: JAXArray | float
) -> JAXArray:
    """
    Return a function that computes DRW Power Spectral Density (PSD).

    Args:
        tau (float): DRW decorrelation/characteristic timescale
        amp (float): DRW RMS amplitude

    Returns:
        A function that takes in frequencies and returns PSD at the given frequencies.
    """

    # convert amp, tau to CARMA parameters
    a0 = 1 / tau
    sigma2 = 2 * amp**2 * a0

    return sigma2 / (a0**2 + (2 * jnp.pi * f) ** 2)


@jax.jit
def drw_acf(t: JAXArray | NDArray, tau: JAXArray | float) -> JAXArray:
    """
    Return a function that computes the DRW autocorrelation function (ACF).

    Args:
        tau (float): DRW decorrelation/characteristic timescale.

    Returns:
        A function that takes in time lags and returns ACF at the given lags.
    """
    # convert to CARMA parameter
    a0 = 1 / tau
    return jnp.exp(-a0 * t)


@jax.jit
def drw_sf(
    t: JAXArray | NDArray, tau: JAXArray | float, amp: JAXArray | float
) -> JAXArray:
    """
    Return a function that computes the structure function (SF) of DRW.

    Args:
        amp (float): DRW RMS amplitude
        tau (float): DRW decorrelation/characteristic timescale.

    Returns:
        A function that takes in time lags and returns DRW SF at the given lags.
    """

    return jnp.sqrt(2 * amp**2 * (1 - drw_acf(t, tau)))
