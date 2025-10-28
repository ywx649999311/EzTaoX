[![DOI](https://zenodo.org/badge/928658755.svg)](https://doi.org/10.5281/zenodo.17467662)
## EzTao*X*
`EzTaoX` is a flexible framework for multi-wavelength and multi-survey AGN light-curve modeling using Gaussian Processes (GPs).  Built on top of `tinygp`---a scalable GP library in `JAX`---`EzTaoX` is fast, modular, and integrates seamlessly with the `JAX` ecosystem for statistical inference and modern machine learning.

> #### `EzTaoX` is under active development, breaking API changes are expected.

### Installation
```
pip install git+https://github.com/ywx649999311/EzTaoX.git
```
#### Dependencies
##### Supports Python 3.10, 3.11, 3.12
```
"jax (<=0.4.31)",
"jaxlib (<=0.4.31)",
"tinygp (>=0.3.0,<0.4.0)",
"jaxopt (>=0.8.3,<0.9.0)",
"optax (>=0.2.4,<0.3.0)",
"numpyro (>=0.17.0,<0.20.0)",
```

### Documentation
Please see tutorials in the `tutorials` folder.


### Acknowledgment
`EzTaoX` is built on top of (and inspired by) [`tinygp`](https://github.com/dfm/tinygp)---a general purpose GP modeling framework written in `JAX`. For more general GP modeling tasks, experienced users can directly explore `tinygp`.
