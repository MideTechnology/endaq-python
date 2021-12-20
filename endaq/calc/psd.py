# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
import warnings

import numpy as np
import pandas as pd
import scipy.signal

from endaq.calc import utils


def _np_histogram_nd(array, bins=10, weights=None, axis=-1, **kwargs):
    """Compute histograms over a specified axis."""
    array = np.asarray(array)
    bins = np.asarray(bins)
    weights = np.asarray(weights) if weights is not None else None

    # Collect all ND inputs
    nd_params = {}
    (nd_params if array.ndim > 1 else kwargs)["a"] = array
    (nd_params if bins.ndim > 1 else kwargs)["bins"] = bins
    if weights is not None:
        (nd_params if weights.ndim > 1 else kwargs)["weights"] = weights

    if len(nd_params) == 0:
        return np.histogram(**kwargs)[0]

    # Move the desired axes to the back
    for k, v in nd_params.items():
        nd_params[k] = np.moveaxis(v, axis, -1)

    # Broadcast ND arrays to the same shape
    ks, vs = zip(*nd_params.items())
    vs_broadcasted = np.broadcast_arrays(*vs)
    for k, v in zip(ks, vs_broadcasted):
        nd_params[k] = v

    # Generate output
    bins = nd_params.get("bins", bins)

    result_shape = ()
    if len(nd_params) != 0:
        result_shape = v.shape[:-1]
    if bins.ndim >= 1:
        result_shape = result_shape + (bins.shape[-1] - 1,)
    else:
        result_shape = result_shape + (bins,)

    result = np.empty(
        result_shape,
        dtype=(weights.dtype if weights is not None else int),
    )
    loop_shape = v.shape[:-1]

    for nd_i in np.ndindex(*loop_shape):
        nd_kwargs = {k: v[nd_i] for k, v in nd_params.items()}
        result[nd_i], edges = np.histogram(**nd_kwargs, **kwargs)

    result = np.moveaxis(result, -1, axis)
    return result


def welch(
    df: pd.DataFrame,
    bin_width: float = 1.0,
    scaling: typing.Literal[None, "density", "spectrum", "parseval"] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Perform `scipy.signal.welch` with a specified frequency spacing.

    :param df: the input data
    :param bin_width: the desired width of the resulting frequency bins, in Hz;
        defaults to 1 Hz
    :param scaling: the scaling of the output; `"density"` & `"spectrum"`
        correspond to the same options in ``scipy.signal.welch``; `"parseval"`
        will maintain the "energy" between the input & output, s.t.
        ``welch(df, scaling="parseval").sum(axis="rows")`` is roughly equal to
        ``df.abs().pow(2).sum(axis="rows")``
    :param kwargs: other parameters to pass directly to ``scipy.signal.welch``
    :return: a periodogram

    .. seealso::

        - `SciPy Welch's method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>`_
          Documentation for the periodogram function wrapped internally.
        - `Parseval's Theorem <https://en.wikipedia.org/wiki/Parseval's_theorem>`_
          - the theorem relating the RMS of a time-domain signal to that of its
          frequency spectrum
    """
    dt = utils.sample_spacing(df)
    fs = 1 / dt

    if scaling == "parseval":
        kwargs["scaling"] = "density"
    elif scaling is not None:
        kwargs["scaling"] = scaling

    freqs, psd = scipy.signal.welch(
        df.values, fs=fs, nperseg=int(fs / bin_width), **kwargs, axis=0
    )
    if scaling == "parseval":
        psd = psd * freqs[1]

    return pd.DataFrame(
        psd, index=pd.Series(freqs, name="frequency (Hz)"), columns=df.columns
    )


def differentiate(df: pd.DataFrame, n: float = 1.0) -> pd.DataFrame:
    """
    Perform time-domain differentiation on periodogram data.

    :param df: a periodogram
    :param n: the time derivative order; negative orders represent integration
    :return: a periodogram of the time-derivated data
    """
    # Involves a division by zero for n < 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        factor = (2 * np.pi * df.index.values) ** (2 * n)  # divide by zero
    if n < 0:
        factor[df.index == 0] = 0

    return df * factor[..., np.newaxis]


def to_jagged(
    df: pd.DataFrame,
    freq_splits: np.array,
    agg: typing.Union[
        typing.Literal["mean", "sum"],
        typing.Callable[[np.ndarray, int], float],
    ] = "mean",
) -> pd.DataFrame:
    """
    Calculate a periodogram over non-uniformly spaced frequency bins.

    :param df: the returned values from ``endaq.calc.psd.welch``
    :param freq_splits: the boundaries of the frequency bins; must be strictly
        increasing
    :param agg: the method for aggregating values into bins; `'mean'` preserves
        the PSD's area-under-the-curve, `'sum'` preserves the PSD's "energy"
    :return: a periodogram with the given frequency spacing
    """
    freq_splits = np.asarray(freq_splits)
    if len(freq_splits) < 2:
        raise ValueError("need at least two frequency bounds")
    if not np.all(freq_splits[:-1] <= freq_splits[1:]):
        raise ValueError("frequency bounds must be strictly increasing")

    # Check that PSD samples do not skip any frequency bins
    spacing_test = np.diff(np.searchsorted(freq_splits, df.index))
    if np.any(spacing_test < 1):
        warnings.warn(
            "empty frequency bins in re-binned PSD; "
            "original PSD's frequency spacing is too coarse",
            RuntimeWarning,
        )

    if isinstance(agg, str):
        if agg not in ("sum", "mean"):
            raise ValueError(f'invalid aggregation mode "{agg}"')

        # Reshape frequencies for histogram function
        f_ndim = np.broadcast_to(df.index.to_numpy()[..., np.newaxis], df.shape)

        # Calculate sum via histogram function
        psd_jagged = _np_histogram_nd(f_ndim, bins=freq_splits, weights=df, axis=0)

        # Adjust values for mean calculation
        if agg == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # x/0

                psd_jagged = np.nan_to_num(  # <- fix divisions by zero
                    psd_jagged
                    / np.histogram(df.index, bins=freq_splits)[0][..., np.newaxis]
                )

    else:
        psd_binned = np.split(df, np.searchsorted(df.index, freq_splits), axis=0)[1:-1]
        psd_jagged = np.stack([agg(a, axis=0) for a in psd_binned], axis=0)

    return pd.DataFrame(
        psd_jagged,
        index=pd.Series((freq_splits[1:] + freq_splits[:-1]) / 2, name=df.index.name),
        columns=df.columns,
    )


def to_octave(
    df: pd.DataFrame, fstart: float = 1.0, octave_bins: float = 12.0, **kwargs
) -> pd.DataFrame:
    """
    Calculate a periodogram over log-spaced frequency bins.

    :param df: the returned values from ``endaq.calc.psd.welch``
    :param fstart: the first frequency bin, in Hz; defaults to 1 Hz
    :param octave_bins: the number of frequency bins in each octave; defaults
        to 12
    :param kwargs: other parameters to pass directly to ``to_jagged``
    :return: a periodogram with the given logarithmic frequency spacing
    """
    max_f = df.index.max()

    octave_step = 1 / octave_bins
    center_freqs = 2 ** np.arange(
        np.log2(fstart),
        np.log2(max_f) + octave_step / 2,
        octave_step,
    )
    freq_splits = 2 ** np.arange(
        np.log2(fstart) - octave_step / 2,
        np.log2(max_f) + octave_step,
        octave_step,
    )
    assert len(center_freqs) + 1 == len(freq_splits)

    result = to_jagged(df, freq_splits=freq_splits, **kwargs)
    result.index = pd.Series(center_freqs, name=df.index.name)
    return result


def vc_curves(
    accel_psd: pd.DataFrame, fstart: float = 1.0, octave_bins: float = 12.0
) -> pd.DataFrame:
    """
    Calculate Vibration Criterion (VC) curves from an acceration periodogram.

    :param accel_psd: a periodogram of the input acceleration
    :param fstart: the first frequency bin
    :param octave_bins: the number of frequency bins in each octave; defaults
        to 12
    :return: the Vibration Criterion (VC) curve of the input acceleration
    """
    """
    # Theory behind the calculation

    Let x(t) be a real-valued time-domain signal, and X(2πf) = F{x(t)}(2πf)
    be the Fourier Transform of that signal. By Parseval's Theorem,

        ∫x(t)^2 dt = ∫|X(2πf)|^2 df

    (see https://en.wikipedia.org/wiki/Parseval%27s_theorem#Notation_used_in_physics)

    Rewriting the right side of that equation in the discrete form becomes

        ∫x(t)^2 dt ≈ ∑ |X[k]|^2 • ∆f

    where ∆f = fs/N = (1/∆t) / N = 1/T.
    Limiting the right side to a range of discrete frequencies (k_0, k_1):

        ∫x(t)^2 dt ≈ [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f

    The VC curve calculation is the RMS over the time-domain. If T is the
    duration of the time-domain signal, then:

        √((1/T) ∫x(t)^2 dt)
            ≈ √((1/T) [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f)
            = ∆f • √([∑; k=k_0 -> k≤k_1] |X[k]|^2)

    If the time-series data is acceleration, then the signal needs to first
    be integrated into velocity. This can be done in the frequency domain
    by replacing |X(2πf)|^2 with (1/2πf)^2 |X(2πf)|^2.
    """
    df_vel = differentiate(accel_psd, n=-1)
    df_vel_oct = to_octave(
        df_vel,
        fstart=fstart,  # Hz
        octave_bins=octave_bins,
        agg="sum",
    )

    # The PSD must already scale by ∆f -> need only scale by √∆f?
    # TODO make `density` parameter, scale differently depending on mode
    return np.sqrt(accel_psd.index[1]) * df_vel_oct.apply(np.sqrt)
