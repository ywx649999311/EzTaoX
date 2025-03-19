"Kernels module"
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import tinygp
from tinygp.helpers import JAXArray


class MultibandLowRank(tinygp.kernels.Kernel):
    amplitudes: jnp.ndarray
    kernel: tinygp.kernels.Kernel

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def evaluate(self, X1, X2) -> JAXArray:
        return (
            self.amplitudes[X1[1]]
            * self.amplitudes[X2[1]]
            * self.kernel.evaluate(X1[0], X2[0])
        )


class MultibandFullRank(tinygp.kernels.Kernel):
    core_kernel: tinygp.kernels.Kernel
    band_kernel: jnp.ndarray

    def __init__(self, kernel, diagonal, off_diagonal) -> None:
        ndim = diagonal.size
        if off_diagonal.size != ((ndim - 1) * ndim) // 2:
            raise ValueError(
                "Dimension mismatch: expected "
                f"(ndim-1)*ndim/2 = {((ndim - 1) * ndim) // 2} elements in "
                f"'off_diagonal'; got {off_diagonal.size}"
            )
        # Construct the band kernel
        factor = jnp.zeros((ndim, ndim))
        factor = factor.at[jnp.diag_indices(ndim)].add(diagonal)
        factor = factor.at[jnp.tril_indices(ndim, -1)].add(off_diagonal)
        self.band_kernel = factor @ factor.T
        self.core_kernel = kernel

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def evaluate(self, X1, X2) -> JAXArray:
        t1, b1 = X1
        t2, b2 = X2

        return self.band_kernel[b1, b2] * self.core_kernel.evaluate(X1, X2)


class MultibandFFT(tinygp.kernels.Kernel):
    amplitudes: jnp.ndarray
    kernel: tinygp.kernels.Kernel
    transfer_function: Callable
    transfer_function_params: dict[str, JAXArray]

    def __init__(self, amplitudes, kernel, transfer_function, **kwargs) -> None:
        self.amplitudes = amplitudes
        self.kernel = kernel
        self.transfer_function = transfer_function
        self.transfer_function_params = kwargs

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def evaluate(self, X1, X2) -> JAXArray:
        t_eval = jnp.linspace(-1000, 1000, 1000)
        dt = t_eval[1] - t_eval[0]
        kernel_eval = self.kernel(t_eval, jnp.array([0])).T[0]

        t1, b1 = X1
        t2, b2 = X2

        # Testing that this gives the same results as the normal kernel
        # return self.amplitudes[b1] * self.amplitudes[b2] * self.kernel.evaluate(t1, t2)

        # Equation 8 in https://ui.adsabs.harvard.edu/abs/2011ApJ...735...80Z/abstract
        Psi1 = self.transfer_function(t_eval, b1, **self.transfer_function_params)
        Psi2 = self.transfer_function(t_eval, b2, **self.transfer_function_params)
        K1 = dt * jax.scipy.signal.fftconvolve(Psi1, kernel_eval, mode="same")
        K2 = dt * jax.scipy.signal.fftconvolve(Psi2, K1, mode="same")

        return (
            self.amplitudes[b1]
            * self.amplitudes[b2]
            * jnp.interp(t1 - t2, t_eval, kernel_eval)
        )
