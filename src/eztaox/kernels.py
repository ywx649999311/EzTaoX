"Kernels module"

import jax.numpy as jnp
import tinygp
from tinygp.helpers import JAXArray


class mb_kernel(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X) -> JAXArray:
        return X[0]

    def observation_model(self, X) -> JAXArray:
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])
