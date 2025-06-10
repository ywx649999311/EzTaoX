# EzTao*X*

## Installation
```
pip install git+https://github.com/ywx649999311/EzTaoX.git
```
## Dependencies
>```
>"jax (<=0.4.31)",
>"jaxlib (<=0.4.31)",
>"tinygp (>=0.3.0,<0.4.0)",
>"jaxopt (>=0.8.3,<0.9.0)",
>"optax (>=0.2.4,<0.3.0)",
>"numpyro (>=0.17.0,<0.20.0)",
>```
## How to run tests
1. Install `pyenv` for managing multiple python versions
2. Install `poetry` managing depencies
3. Use `pyenv` to install multiple python versions with:
```
pyenv install 3.10 3.11 3.12 3.13
```
1. Activate multiple python versions
```
pyenv shell 3.10 3.11 3.12 3.13
```
1. Run tests
```
nox -s tests
```
