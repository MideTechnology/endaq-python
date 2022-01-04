from __future__ import annotations

import typing
from typing import Optional

import numpy as np
import pandas as pd
import scipy.fft

from endaq.calc import psd, utils


def aggregate_fft(df, **kwargs):
    """
    Calculate the FFT using `scipy.signal.welch` with a specified frequency spacing.  The data returned is in the same
    units as the data input.

    :param df: the input data
    :param bin_width: the desired width of the resulting frequency bins, in Hz;
        defaults to 1 Hz
    :param kwargs: other parameters to pass directly to ``scipy.signal.welch``
    :return: a periodogram

    .. seealso::

        - `SciPy Welch's method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_
          Documentation for the periodogram function wrapped internally.
    """
    kwargs['scaling'] = 'unit'
    kwargs['noverlap'] = 0
    return psd.welch(df, **kwargs)


def fft(
        df: pd.DataFrame,
        output: typing.Literal[None, "magnitude", "angle", "complex"] = None,
        nfft: Optional[int] = None,
        norm: typing.Literal[None, "unit", "forward", "ortho", "backward"] = None,
    ) -> pd.DataFrame:
    """
    Perform the FFT of the data in `df`, using Scipy's FFT method from `scipy.fft.fft`.

    :param df: the input data
    :param output: *Optional*  The type of the output of the FFT. Default is "magnitude".  "magnitude" will return the
                   absolute value of the FFT, "angle" will return the phase angle in radians, "complex" will return the
                   complex values of the FFT.
    :param nfft: *Optional* Length of the transformed axis of the output. If nfft is smaller than the length of the
                 input, the input is cropped. If it is larger, the input is padded with zeros. If n is not given, the
                 length of the input along the axis specified by axis is used.
    :param norm: *Optional* Normalization mode. Default is “forward”, meaning a normalization of 1/n is applied on the
                 forward transforms and no normalization is applied on the ifft. “backward” instead applies no
                 normalization on the forward tranform and 1/n on the backward. For norm="ortho", both directions are
                 scaled by 1/sqrt(n).
    :param kwargs: Further keywords passed to `scipy.fft.fft`.  Note that the nfft parameter of this function is passed
                   to `scipy.fft.fft` as `n`.
    :return: The FFT of each channel in `df`.

    .. seealso::

        - `SciPy FFT method <https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.fft.fft.html>`_
          Documentation for the FFT function wrapped internally.
    """

    if output is None:
        output_fun = np.abs
    elif output not in ["magnitude", "angle", "complex"]:
        raise ValueError(f'output must be one of None, "magnitude", "angle", or "complex".  Was {output}.')
    else:
        if output == "magnitude":
            output_fun = np.abs
        elif output == "angle":
            output_fun = np.angle
        elif output == "complex":
            output_fun = np.asarray
        else:
            raise Exception("impossible state reached in fft output type")

    if nfft is None:
        nfft = len(df)
    elif not isinstance(nfft, int):
        raise TypeError(f'nfft must be a positive integer, was of type {type(nfft)}')
    elif nfft <= 0:
        raise ValueError(f'nfft must be positive, was {nfft}')

    if norm is None:
        norm = "forward"
    elif norm not in ["forward", "ortho", "backward"]:
        raise ValueError(f'norm must be one of None, "forward", "ortho", or "backward".  Was {norm}.')

    dt = utils.sample_spacing(df)
    fs = 1 / dt

    return pd.DataFrame(
        data={
            c: output_fun(scipy.fft.fftshift(scipy.fft.fft(
                df[c].to_numpy(),
                n=nfft,
                norm=norm,
            )))
            for c in df.columns},
        columns=df.columns,
        index=scipy.fft.fftshift(scipy.fft.fftfreq(nfft))*fs,
    )


def rfft():
    pass


def dct():
    pass
