"""Parameter Initializers."""

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from tinygp.helpers import JAXArray


class InitializerBase(eqx.Module):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, key: jax.random.PRNGKey, nSample: int) -> JAXArray:
        raise NotImplementedError


class UniformInit(InitializerBase):
    n: int
    Range: Sequence[JAXArray | float]

    def __init__(self, n, Range) -> None:
        self.n = n
        self.Range = Range

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSample: int,
    ) -> JAXArray:
        key1, _ = jax.random.split(key, 2)
        mean = dist.Uniform(*self.Range).sample(key1, (nSample, self.n))
        return mean


class ExpInit(InitializerBase):
    logScaleRange: Sequence[JAXArray | float]
    logSigmaRange: Sequence[JAXArray | float]

    def __init__(
        self,
        logScaleRange,
        logSigmaRange,
    ) -> None:
        self.logScaleRange = logScaleRange
        self.logSigmaRange = logSigmaRange

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSample: int,
    ) -> JAXArray:
        key1, key2 = jax.random.split(key)
        scale = dist.Uniform(*self.logScaleRange).sample(key1, (nSample,))
        sigma = dist.Uniform(*self.logSigmaRange).sample(key2, (nSample,))

        return jnp.stack([scale, sigma], axis=-1)


class CeleriteInit(InitializerBase):
    aRange: Sequence[JAXArray | float]
    bRange: Sequence[JAXArray | float]
    cRange: Sequence[JAXArray | float]
    dRange: Sequence[JAXArray | float]

    def __init__(
        self,
        aRange,
        bRange,
        cRange,
        dRange,
    ) -> None:
        self.aRange = aRange
        self.bRange = bRange
        self.cRange = cRange
        self.dRange = dRange

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSample: int,
    ) -> JAXArray:
        key1, key2, key3, key4 = jax.random.split(key, 4)
        a = dist.Uniform(*self.aRange).sample(key1, (nSample,))
        b = dist.Uniform(*self.bRange).sample(key2, (nSample,))
        c = dist.Uniform(*self.cRange).sample(key3, (nSample,))
        d = dist.Uniform(*self.dRange).sample(key4, (nSample,))

        return jnp.stack([a, b, c, d], axis=-1)


class SHOInit(InitializerBase):
    logOmegaRange: Sequence[JAXArray | float]
    logQualityRange: Sequence[JAXArray | float]
    logSigmaRange: Sequence[JAXArray | float]

    def __init__(
        self,
        logOmegaRange,
        logQualityRange,
        logSigmaRange,
    ) -> None:
        self.logOmegaRange = logOmegaRange
        self.logQualityRange = logQualityRange
        self.logSigmaRange = logSigmaRange

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSample: int,
    ) -> JAXArray:
        key1, key2, key3 = jax.random.split(key, 3)
        omega = dist.Uniform(*self.logOmegaRange).sample(key1, (nSample,))
        quality = dist.Uniform(*self.logQualityRange).sample(key2, (nSample,))
        sigma = dist.Uniform(*self.logSigmaRange).sample(key3, (nSample,))

        return jnp.stack([omega, quality, sigma], axis=-1)


# alias
DRWInit = ExpInit
Matern32Init = ExpInit
Matern52Init = ExpInit
CosineInit = ExpInit
