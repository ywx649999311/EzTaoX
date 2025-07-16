## EzTao*X*
`EzTaoX` is designed to perform multi-wavelength/multi-band modeling of AGN light curves using Gaussian Processes (GPs). `EzTaoX` aims to establish a new framework that can utilize the multi-band coverage of modern time-domain surveys to simultaneously extract stochastic variability and inter-band time delay information from multi-band light curves, and at the same time provide significant improvement in computational speed in comparison to existing tools of similar capabilities. 

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


### Credits
`EzTaoX` is built on top of (and inspired by) [`tinygp`](https://github.com/dfm/tinygp)---a general purpose GP modeling framework written in `JAX`. For more general GP modeling tasks, pro users can directly explore `tinygp`. 
