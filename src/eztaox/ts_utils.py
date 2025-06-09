import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from tinygp.helpers import JAXArray


@jax.jit
def _get_nearest_idx(tIn, x) -> int:
    """
    Get the index of the nearest value in `tIn` to `x`.

    Args:
        tIn (JAXArray): Array of time values.
        x (float): The value to find the nearest index for.
    """
    return jnp.argmin(jnp.abs(tIn - x))


def downsampleByTime(tIn, tOut) -> JAXArray:
    """
    Downsample `tIn` to match the time points in `tOut`.

    Args:
        tIn (JAXArray): Array of time values to be downsampled.
        tOut (JAXArray): Array of target time values.

    Returns:
        JAXArray: Downsampled array of time values.
    """
    return tIn[jax.vmap(_get_nearest_idx, in_axes=(None, 0))(tIn, tOut)]


def formatlc(
    ts: dict[str, NDArray | JAXArray],
    ys: dict[str, NDArray | JAXArray],
    yerrs: dict[str, NDArray | JAXArray],
    band_order: dict[str, int],
) -> tuple[tuple[JAXArray, JAXArray], JAXArray, JAXArray]:
    "Transform data in dictionary to TinyGP friendly format."

    band_keys = band_order.keys()
    tss = jnp.concat([ts[key] for key in band_keys])
    yss = jnp.concat([ys[key] for key in band_keys])
    yerrss = jnp.concat([yerrs[key] for key in band_keys])
    band_idxs = jnp.concat(
        [jnp.ones(len(ts[x])) * band_order[x] for x in band_keys]
    ).astype(int)

    return (tss, band_idxs), yss, yerrss
