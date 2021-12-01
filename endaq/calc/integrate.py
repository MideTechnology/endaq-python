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
    offset_mode: Union[typing.Literal[None, "mean", "median"], Callable] = None,
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
    if offset_mode is None:
        return result

    if offset_mode in ("mean", "median"):
        offset_mode = dict(mean=np.mean, median=np.median)[offset_mode]

    return result - result.apply(offset_mode, axis="index", raw=True)


def iter_integrals(
    df: pd.DataFrame,
    offset_mode: Union[typing.Literal[None, "mean", "median"], Callable] = None,
    highpass_cutoff: Optional[float] = None,
    filter_half_order: int = 3,
    tukey_percent: float = 0.0,
) -> Iterable[pd.DataFrame]:
    """
    Iterate over conditioned integrals of the given original data.

    :param df: the input data
    :param highpass_cutoff: the cutoff frequency of a preconditioning highpass
        filter; if None, no filter is applied
    :param filter_half_order: the half-order of the preconditioning highpass
        filter, if used
    :param tukey_percent: the alpha parameter of a preconditioning tukey filter;
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
            half_order=filter_half_order,
            low_cutoff=highpass_cutoff,
            high_cutoff=None,
            tukey_percent=tukey_percent,
        )
        df = _integrate(df, offset_mode)


def integrals(
    df: pd.DataFrame,
    n: int = 1,
    offset_mode: Union[typing.Literal[None, "mean", "median"], Callable] = None,
    highpass_cutoff: Optional[float] = None,
    tukey_percent: float = 0.0,
) -> List[pd.DataFrame]:
    """
    Calculate `n` integrations of the given data.

    :param df: the data to integrate, indexed with timestamps
    :param n: the number of integrals to calculate
    :param highpass_cutoff: the cutoff frequency for the initial highpass filter;
        this is used to remove artifacts caused by DC trends
    :param tukey_percent: the alpha parameter of a preconditioning tukey filter;
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
                offset_mode=offset_mode,
                highpass_cutoff=highpass_cutoff,
                filter_half_order=n // 2 + 1,  # ensures zero DC content in nth integral
                tukey_percent=tukey_percent,
            ),
        )
    ]
