# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
import scipy.signal


def sample_spacing(
    data: Union[np.ndarray, pd.DataFrame],
    convert: typing.Literal[None, "to_seconds"] = "to_seconds",
) -> Union[float, np.timedelta64]:
    """
    Calculate the average spacing between individual samples.

    For time indices, this calculates the sampling period `dt`.

    :param data: the input data; either a pandas DataFrame with the samples
        spaced along its index, or a 1D-array-like of sample times
    :param convert: if `"to_seconds"` (default), convert any time objects into
        floating-point seconds
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.index

    dt = (data[-1] - data[0]) / (len(data) - 1)
    if convert == "to_seconds" and isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    return dt


def logfreqs(
    df: pd.DataFrame, init_freq: Optional[float] = None, bins_per_octave: float = 12.0
) -> np.ndarray:
    """
    Calculate a sequence of log-spaced frequencies for a given dataframe.

    :param df: the input data
    :param init_freq: the initial frequency in the sequence; if `None` (default),
        use the frequency corresponding to the data's duration
    :param bins_per_octave: the number of frequencies per octave
    :return: an array of log-spaced frequencies
    """
    dt = sample_spacing(df)
    T = dt * len(df.index)

    if init_freq is None:
        init_freq = 1 / T
    elif 1 / init_freq > T:
        warnings.warn(
            "the data's duration is too short to accurately represent an"
            f" initial frequency of {init_freq:.3f} Hz",
            RuntimeWarning,
        )

    return 2 ** np.arange(
        np.log2(init_freq),
        np.log2(1 / dt) - 1,
        1 / bins_per_octave,
    )


def to_dB(data: np.ndarray, reference: float, squared: bool = False) -> np.ndarray:
    """
    Scale data into units of decibels.

    Decibels are a log-scaled ratio of some value against a reference;
    typically this is expressed as follows:

    .. math:: dB = 10 \\log10\\left( \\frac{x}{x_{\\text{ref}}} \\right)

    By convention, "decibel" units tend to operate on units of *power*. For
    units that are proportional to power *when squared* (e.g., volts, amps,
    pressure, etc.), their "decibel" representation is typically doubled (i.e.,
    :math:`dB = 20 \\log10(...)`). Users can specify which scaling to use
    with the `squared` parameter.

    .. note::
        Decibels can **NOT** be calculated from negative values.

        For example, to calculate dB on arbitrary time-series data, typically
        data is first aggregated via a total or a rolling RMS or PSD, and the
        non-negative result is then scaled into decibels.

    :param data: the input data
    :param reference: the reference value corresponding to 0dB
    :param squared: whether the input data & reference value are pre-squared;
        defaults to `False`

    .. seealso::
        - ``endaq.calc.stats.rms``
        - ``endaq.calc.stats.rolling_rms``
        - ``endaq.calc.psd.welch``
    """
    if reference <= 0:
        raise ValueError("reference value must be strictly positive")

    data = np.asarray(data)
    if np.any(data < 0):
        raise ValueError(
            "cannot compute decibels from negative values (see the docstring"
            " for details)"
        )

    return (10 if squared else 20) * (np.log10(data) - np.log10(reference))


dB_refs = {
    "SPL": 2e-5,  # Pascal
    "audio_intensity": 1e-12,  # W/m²
}


def resample(df: pd.DataFrame, sample_rate: Optional[float] = None) -> pd.DataFrame:
    """
    Resample a dataframe to a desired sample rate (in Hz)

    :param df: The DataFrame to resample, indexed by time
    :param sample_rate: The desired sample rate to resample the given data to.
     If one is not supplied, then it will use the same as it currently does, but
     make the time stamps uniformally spaced
    :return: The resampled data in a DataFrame
    """
    if sample_rate is None:
        num_samples_after_resampling = len(df)
    else:
        dt = sample_spacing(df)
        num_samples_after_resampling = int(dt * len(df) * sample_rate)

    resampled_data, resampled_time = scipy.signal.resample(
        df,
        num_samples_after_resampling,
        t=df.index.values.astype('datetime64[s]'),
    )
    resampled_df = pd.DataFrame(
        resampled_data,
        index=resampled_time,
        columns=df.columns,
    )
    return resampled_df
