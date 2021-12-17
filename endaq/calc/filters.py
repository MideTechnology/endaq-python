from __future__ import annotations

from typing import Optional, Union, Tuple
import functools

import pandas as pd
import scipy.signal

from endaq.calc import utils


def butterworth(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Butterworth filter to an array.

    This function uses Butterworth filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :return: the filtered data

    .. seealso::

        - `SciPy Butterworth filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
          Documentation for the butterworth filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs: Union[float, Tuple[float, float]]
    filter_type: str

    if low_cutoff is not None and high_cutoff is not None:
        cutoff_freqs = (low_cutoff, high_cutoff)
        filter_type = "bandpass"
    elif low_cutoff is not None:
        cutoff_freqs = low_cutoff
        filter_type = "highpass"
    elif high_cutoff is not None:
        cutoff_freqs = high_cutoff
        filter_type = "lowpass"
    else:
        filter_type = ""

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.butter(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df
