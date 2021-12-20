from __future__ import annotations

import typing
from typing import List, Optional, Iterable, Union, Callable
import functools

import numpy as np
import pandas as pd
import scipy.integrate

from endaq.calc import filters, utils


def _integrate(
    df: pd.DataFrame,
    zero: Union[typing.Literal["start", "mean", "median"], Callable] = "start",
) -> pd.DataFrame:
    """Integrate data over an axis."""
    dt = utils.sample_spacing(df)

    result = df.apply(
        functools.partial(scipy.integrate.cumulative_trapezoid, dx=dt, initial=0),
        axis=0,
        raw=True,
    )
    # In lieu of explicit initial offset, set integration bias to remove mean
    # -> avoids trend artifacts after successive integrations
    if zero == "start":
        return result

    if zero in ("mean", "median"):
        zero = dict(mean=np.mean, median=np.median)[zero]

    return result - result.apply(zero, axis="index", raw=True)


def iter_integrals(
    df: pd.DataFrame,
    zero: Union[typing.Literal["start", "mean", "median"], Callable] = "start",
    highpass_cutoff: Optional[float] = None,
    tukey_percent: float = 0.0,
) -> Iterable[pd.DataFrame]:
    """
    Iterate over conditioned integrals of the given original data.

    :param df: the input data
    :param zero: the output quantity driven to zero by the integration constant;
        `"start"` (default) chooses an integration constant of ``-output[0]``,
        `"mean"` chooses ``-np.mean(output)`` & `"median"` chooses
        ``-np.median(output)``
    :param highpass_cutoff: the cutoff frequency of a preconditioning highpass
        filter; if None, no filter is applied
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :return: an iterable over the data's successive integrals; the first item
        is the preconditioned input data

    .. seealso::

        - `SciPy trapezoid integration <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html>`_
          Documentation for the integration function used internally.

        - `SciPy Butterworth filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
          Documentation for the butterworth filter design function used in
          preprocessing.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    while True:
        yield df.copy()  # otherwise, edits to the yielded item would alter the results
        df = filters.butterworth(
            df,
            low_cutoff=highpass_cutoff,
            high_cutoff=None,
            tukey_percent=tukey_percent,
        )
        df = _integrate(df, zero)


def integrals(
    df: pd.DataFrame,
    n: int = 1,
    zero: Union[typing.Literal["start", "mean", "median"], Callable] = "start",
    highpass_cutoff: Optional[float] = None,
    tukey_percent: float = 0.0,
) -> List[pd.DataFrame]:
    """
    Calculate `n` integrations of the given data.

    :param df: the data to integrate, indexed with timestamps
    :param n: the number of integrals to calculate; defaults to `1`
    :param zero: the output quantity driven to zero by the integration constant;
        `"start"` (default) chooses an integration constant of ``-output[0]``,
        `"mean"` chooses ``-np.mean(output)`` & `"median"` chooses
        ``-np.median(output)``
    :param highpass_cutoff: the cutoff frequency for the initial highpass filter;
        this is used to remove artifacts caused by DC trends
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :return: a length `n+1` list of the kth-order integrals from 0 to n (inclusive)

    .. seealso::

        - `SciPy trapezoid integration <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html>`_
          Documentation for the integration function used internally.

        - `SciPy Butterworth filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
          Documentation for the butterworth filter design function used in
          preprocessing.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    return [
        integ
        for _, integ in zip(
            range(n + 1),
            iter_integrals(
                df,
                zero=zero,
                highpass_cutoff=highpass_cutoff,
                tukey_percent=tukey_percent,
            ),
        )
    ]
