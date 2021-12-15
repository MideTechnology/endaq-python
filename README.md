# `endaq-python`: A comprehensive, user-centric Python API for working with enDAQ data and devices


## Installation

endaq is available on PYPI via `pip`:

    pip install endaq

For the most recent features that are still under development, you can also use `pip` to install endaq directly from GitHub:

    pip install git+https://github.com/MideTechnology/endaq-python.git@development

## Contents

This package consists of several submodules, you can read more about how these are used in their respective readme files:
* `endaq.calc` ([readme](https://github.com/MideTechnology/endaq-python/tree/main/endaq/calc)): A computational backend for vibration analysis.
* `endaq.cloud` ([readme](https://github.com/MideTechnology/endaq-python/tree/main/endaq/cloud)): Tools for interacting with enDAQ Cloud services.
* `endaq.ide` ([readme](https://github.com/MideTechnology/endaq-python/tree/main/endaq/ide)): High-level utility functions to aid in importing and inspecting enDAQ IDE recording files.
* `endaq.plot` ([readme](https://github.com/MideTechnology/endaq-python/tree/main/endaq/plot)):  A package comprising a collection of plotting utilities for sensor data analysis.

## Docs

The docs for this package can be found [here](https://docs.endaq.com/en/latest/).

To locally build the [Sphinx](https://www.sphinx-doc.org) documentation from a clone of the repo:
1. `cd <repo root dir>`
2. `pip install -e .[docs]`
3. `sphinx-build -W -b html docs docs/_build`

## License 

The endaq-python repository is licensed under the MIT license. The full text can be found in the [LICENSE file](https://github.com/MideTechnology/endaq-python/blob/main/LICENSE).
