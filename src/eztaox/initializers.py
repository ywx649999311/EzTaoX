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
    def __call__(self, key: jax.random.PRNGKey, nSamples: int) -> JAXArray:
        raise NotImplementedError


class ExpInit(InitializerBase):
    scaleRange: Sequence[JAXArray | float]
    sigmaRange: Sequence[JAXArray | float]

    def __init__(
        self,
        scaleRange,
        sigmaRange,
    ) -> None:
        self.scaleRange = scaleRange
        self.sigmaRange = sigmaRange

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSamples: int,
    ) -> JAXArray:
        key1, key2 = jax.random.split(key)
        scale = dist.Uniform(*self.scaleRange).sample(key1, (nSamples,))
        sigma = dist.Uniform(*self.sigmaRange).sample(key2, (nSamples,))

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
        nSamples: int,
    ) -> JAXArray:
        key1, key2, key3, key4 = jax.random.split(key, 4)
        a = dist.Uniform(*self.aRange).sample(key1, (nSamples,))
        b = dist.Uniform(*self.bRange).sample(key2, (nSamples,))
        c = dist.Uniform(*self.cRange).sample(key3, (nSamples,))
        d = dist.Uniform(*self.dRange).sample(key4, (nSamples,))

        return jnp.stack([a, b, c, d], axis=-1)


class SHOInit(InitializerBase):
    omegaRange: Sequence[JAXArray | float]
    qualityRange: Sequence[JAXArray | float]
    sigmaRange: Sequence[JAXArray | float]

    def __init__(
        self,
        omegaRange,
        qualityRange,
        sigmaRange,
    ) -> None:
        self.omegaRange = omegaRange
        self.qualityRange = qualityRange
        self.sigmaRange = sigmaRange

    def __call__(
        self,
        key: jax.random.PRNGKey,
        nSamples: int,
    ) -> JAXArray:
        key1, key2, key3 = jax.random.split(key, 3)
        omega = dist.Uniform(*self.omegaRange).sample(key1, (nSamples,))
        quality = dist.Uniform(*self.qualityRange).sample(key2, (nSamples,))
        sigma = dist.Uniform(*self.sigmaRange).sample(key3, (nSamples,))

        return jnp.stack([omega, quality, sigma], axis=-1)


# alias
DRWInit = ExpInit
Matern32Init = ExpInit
Matern52Init = ExpInit
CosineInit = ExpInit
