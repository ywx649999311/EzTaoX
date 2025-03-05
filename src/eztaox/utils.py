from typing import Any

import jax.numpy as jnp
from tinygp.helpers import JAXArray


def formatlc(
    ts: dict[str, Any],
    ys: dict[str, Any],
    yerrs: dict[str, Any],
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
