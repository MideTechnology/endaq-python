from __future__ import annotations

import typing  # for `SupportsIndex`, which is Python3.8+ only
from typing import Union
from collections.abc import Sequence

import numpy as np
import pandas as pd


def L2_norm(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the L2 norm (a.k.a. the Euclidean Norm).

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the L2 norm is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions of size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.sum(np.abs(array) ** 2, axis=axis, keepdims=keepdims))


def max_abs(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the maximum of the absolute value of an array.

    This function should be equivalent to, but generally use less memory than
    ``np.amax(np.abs(array))``.

    Specifically, it generates the absolute-value maximum from ``np.amax(array)``
    and ``-np.amin(array)``. Thus instead of allocating space for the
    intermediate array ``np.abs(array)``, it allocates for the axis-collapsed
    smaller arrays ``np.amax(array)`` & ``np.amin(array)``.

    .. note:: This method does **not** work on complex-valued arrays.

    :param array: the input data
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the absolute maximum is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    # Forbid complex-valued data
    if np.iscomplexobj(array):
        raise ValueError("`max_abs` does not accept complex arrays")

    return np.maximum(
        np.amax(array, initial=-np.inf, axis=axis, keepdims=keepdims),
        -np.amin(array, initial=np.inf, axis=axis, keepdims=keepdims),
    )


def rms(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Calculate the root-mean-square (RMS) along a given axis.

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the RMS is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.mean(np.abs(array) ** 2, axis=axis, keepdims=keepdims))


def rolling_rms(
    df: Union[pd.DataFrame, pd.Series], window_len: int, *args, **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate a rolling root-mean-square (RMS) over a pandas `DataFrame`.

    This function is equivalent to, but computationally faster than the following::

        df.rolling(window_len).apply(endaq.calc.stats.rms)

    :param df: the input data
    :param window_len: the length of the rolling window
    :param args: the positional arguments to pass into ``df.rolling().mean``
    :param kwargs: the keyword arguments to pass into ``df.rolling().mean``
    :return: the rolling-windowed RMS

    .. seealso::
        
        - `Pandas Rolling Mean <https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.mean.html>`_
          - official documentation for ``df.rolling().mean``
        - `Pandas Rolling Standard Deviation method <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.std.html>`_
          - similar to this function, but first removes the windowed mean before squaring
    """
    return df.pow(2).rolling(window_len).mean(*args, **kwargs).apply(np.sqrt, raw=True)
