# EzTaoX

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
poetry run nox -s test
```
