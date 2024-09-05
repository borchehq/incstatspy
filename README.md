# incstatspy

A Python C extension for efficiently calculating running statistics on NumPy arrays. This module provides functions for computing mean, variance, skewness, kurtosis, and central moments.

## Status
![Build Status](https://github.com/borchehq/incstatsPy/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/borchehq/incstatsPy/graph/badge.svg?token=ZSESQKJEKQ)](https://codecov.io/gh/borchehq/incstatsPy)
[![PyPI - Version](https://img.shields.io/pypi/v/incstatspy?label=PyPI&color=green&cacheSeconds=120)](https://pypi.org/project/incstatspy/)


## Features

- Compute running mean, variance, skewness, and kurtosis.
- Calculate central moments up to a specified order.
- Calculate standardized central moments up to a specified order.
- Designed for efficiency and simplicity.
- Handles weighted data points.
- Iterative update functionality when calling on multiple ndarrays of compatible shape.

## Installation

You can install this project using `pip`:

### Using pip

To install the latest version from PyPI:

```bash
pip install incstatspy
```

### From Source

If you want to install the project from the source code:

```bash
git clone https://github.com/borchehq/incstatsPy.git
cd incstatspy
pip install .
```

## Usage
```python
import incstatspy
import numpy as np


# Calculate mean
ndarray = np.array(np.random.rand())
mean, buffer = incstatspy.mean(ndarray)
print(mean)
# Update mean with new data
ndarray = np.array(np.random.rand())
print(mean)
mean, buffer = incstatspy.mean(ndarray, buffer=buffer)
# Calculate variance
ndarray = np.array(np.random.rand())
# Mean will be automatically calculated with variance
mean, variance, buffer = incstatspy.variance(ndarray)
print(variance)
ndarray = np.array(np.random.rand())
# Update mean and variance
mean, variance, buffer = incstatspy.variance(ndarray, buffer=buffer)
print(variance)
ndarray = np.random.rand(3, 3)
# Calculate skewness (variance and mean will be automatically calculated as well)
*_, skewness, buffer = incstatspy.skewness(ndarray) # discard mean and variance
print(skewness)
ndarray = np.random.rand(3, 3)
# Update skewness
*_, skewness, buffer = incstatspy.skewness(ndarray, buffer=buffer)
print(skewness)
ndarray = np.random.rand(3, 3)
# Calculate kurtosis (skewness, variance and mean will be automatically calculated as well)
*_, kurtosis, buffer = incstatspy.kurtosis(ndarray) # discard mean, variance and skewness
print(kurtosis)
ndarray = np.random.rand(3, 3)
# Update kurtosis
*_, kurtosis, buffer = incstatspy.kurtosis(ndarray, buffer=buffer)
print(kurtosis)
# Calculate 8th moment and discard lower moments. Mean will be calculated as well.
# Calculate along axis 1 (default: axis 0)
p = 8
ndarray = np.random.rand(3, 3)
*_, pth_moment, mean, buffer = incstatspy.central_moment(ndarray, p, axis=1)
print(pth_moment)
ndarray = np.random.rand(3, 3)
# Update 8th moment
*_, pth_moment, mean, buffer = incstatspy.central_moment(ndarray, p, axis=1, buffer=buffer)
print(pth_moment)
```

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

The full text of the license is available in the `LICENSE.txt` file in this repository.