# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
import warnings
import datetime

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
        ``df.abs().pow(2).sum(axis="rows")``;
        `"unit"` will maintain the units and scale of the input data.
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
    elif scaling == "unit":
        kwargs["scaling"] = "density"
    elif scaling is not None:
        kwargs["scaling"] = scaling

    nperseg = int(fs / bin_width)
    if "nfft" not in kwargs:
        nfft = nperseg
    else:
        nfft = kwargs["nfft"]

    freqs, psd = scipy.signal.welch(df.values, fs=fs, nperseg=nperseg, **kwargs, axis=0)
    if scaling == "parseval":
        psd = psd * freqs[1]
    elif scaling == "unit":
        psd *= 2 * freqs[1]
        psd *= nfft / nperseg
        psd **= 0.5

    return pd.DataFrame(
        psd, index=pd.Series(freqs, name="frequency (Hz)"), columns=df.columns
    )


def differentiate(df: pd.DataFrame, n: float = 1.0) -> pd.DataFrame:
    """
    Perform time-domain differentiation on periodogram data.

    :param df: a periodogram
    :param n: the time derivative order; negative orders represent integration
    :return: a periodogram of the time-differentiated data
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

    :param df: the returned values from :py:func:`endaq.calc.psd.welch`
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

    :param df: the returned values from :py:func:`endaq.calc.psd.welch`
    :param fstart: the first frequency bin, in Hz; defaults to 1 Hz
    :param octave_bins: the number of frequency bins in each octave; defaults
        to 12
    :param kwargs: other parameters to pass directly to :py:func:`to_jagged`
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


def _aligned_bin_width(fstart: float = 1.0, octave_bins: float = 12.0) -> float:
    """
    Calculate a "good" linearly-spaced bin width, from a PSD of which an
    octave-spaced PSD can be accurately calculated.

    When computing the octave-spaced PSD, linear bins are mapped to octave bins
    and the linear bin values are then summed together within each octave bin.
    However, this calculation can produce an octave-spaced PSD that is an
    *inaccurate* representation of the original linearly-spaced PSD, if the
    mapping from linear to octave bins is *poorly aligned*; e.g., if a
    significant linear bin is halfway between two adjacent octave bins.

    Thus, the main criterion for a "good" linear spacing is how well the bin
    borders in the linearly-spaced PSD align with those of the octave-spaced
    PSD. This is mostly only significant at the lower frequencies, where the
    octave-spaced bins comprise only a handful of linearly-spaced bins.

    :param fstart: the starting frequency of the octave-spaced PSD
    :param octave_bins: the number of bins per octave in the octave-spaced PSD
    :return: a bin-width for a linearly-spaced PSD that aligns well with an
        octave-spaced PSD defined by the given spacing parameters

    .. todo::
        allow user to adjust output resolution:
        - add "good"-ness as a parameter to adjust resolution of result?
        - iterate over solutions with increasing resolutions -> user just picks
          the most appropriate one?
    """
    # To get a good alignment with an octave-spaced frequency bin, the linear
    # bin width should:
    # - be a *"half" divisor* of the octave bin's lower and upper bounds:
    #       f_oct_lower = f_lin * (n1 + 1/2)
    #       f_oct_upper = f_lin * (n2 + 1/2)
    #   for integers n1, n2
    # -> be a divisor of the octave bin's breadth:
    #       f_oct_upper - f_oct_lower = f_lin * (n2 - n1) = f_lin * m
    lbound_factor = 2 ** (-1 / (2 * octave_bins))
    ubound_factor = 2 ** (1 / (2 * octave_bins))
    breadth_factor = ubound_factor - lbound_factor

    # TODO find another way to calculate a linear bin width that better fits
    # the above description
    arb_odd_int = 5  # any odd integer; 5 provides some resolution w/o too many segments
    return fstart / int(arb_odd_int / breadth_factor)


def vc_curves(
    accel_psd: pd.DataFrame, fstart: float = 1.0, octave_bins: float = 12.0
) -> pd.DataFrame:
    """
    Calculate Vibration Criterion (VC) curves from an acceleration periodogram.

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


def rolling_psd(
        df: pd.DataFrame,
        bin_width: float = 1.0,
        octave_bins: float = None,
        fstart: float = 1.0,
        scaling: typing.Literal[None, "density", "spectrum", "parseval", "unit", "rms"] = None,
        agg: typing.Union[
            typing.Literal["mean", "sum"],
            typing.Callable[[np.ndarray, int], float],
        ] = "mean",
        freq_splits: np.array = None,
        num_slices: int = 100,
        indexes: np.array = None,
        index_values: np.array = None,
        slice_width: float = None,
        add_resultant: bool = True,
        disable_warnings: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """
    Compute PSDs for defined slices of a time series data set using :py:func:`~endaq.calc.psd.welch()`
    
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param bin_width: the bin width or resolution in Hz for the PSD, defaults to 1,
        this is ignored if `octave_bins` is defined
    :param octave_bins: the number of frequency bins in each octave, defaults to `None`
    :param fstart: the lowest frequency for an octave PSD, defaults to 1
    :param scaling: the scaling of the output; `"density"` & `"spectrum"`
        correspond to the same options in ``scipy.signal.welch``; `"parseval"`
        will maintain the "energy" between the input & output, s.t.
        ``welch(df, scaling="parseval").sum(axis="rows")`` is roughly equal to
        ``df.abs().pow(2).sum(axis="rows")``;
        `"unit"` will maintain the units and scale of the input data. `"rms"` will use `"parseval"` for the PSD
        calculations and set `agg` to "sum", but then take the square root at the end
    :param agg: the method for aggregating values into bins (only used if converting to octave or jagged); `'mean'` preserves
        the PSD's area-under-the-curve, `'sum'` preserves the PSD's "energy"
    :param freq_splits: the boundaries of the frequency bins to pass to :py:func:`~endaq.calc.psd.to_jagged()`
    :param num_slices: the number of slices to split the time series into, default is 100,
        this is ignored if `indexes` is defined
    :param indexes: the center index locations (not value) of each slice to compute the PSD
    :param index_values: the index values of each peak event to quantify (slower but more intuitive than using `indexes`)
    :param slice_width: the time in seconds to center about each slice index,
        if none is provided it will calculate one based upon the number of slices
    :param add_resultant: if `True` (default) the root sum of squares of each PSD column will
        also be computed
    :param disable_warnings: if `True` (default) it disables the warnings on the PSD length
    :param kwargs: Other parameters to pass directly to :py:func:`psd.welch`
    :return: a dataframe containing all the PSDs, stacked on each other

    See example use cases and syntax at :py:func:`~endaq.plot.spectrum_over_time()`
    which visualizes the output of this function in Heatmaps, Waterfall plots, 
    Surface plots, and Animations

    """
    use_spectrogram = True
    if (indexes is not None) | (index_values is not None):
        use_spectrogram = False

    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df,
        indexes=indexes,
        index_values=index_values,
        num_slices=num_slices,
        slice_width=slice_width
    )

    if use_spectrogram:
        if "window" not in kwargs:
            kwargs["window"] = "hann"
        psd_df = spectrogram(
            df=df,
            bin_width=bin_width,
            num_slices=num_slices,
            scaling=scaling,
            octave_bins=octave_bins,
            fstart=fstart,
            freq_splits=freq_splits,
            add_resultant=add_resultant,
            disable_warnings=disable_warnings,
            **kwargs
        )
    else:
        # Allow for RMS calculations
        exp = 1
        if scaling == "rms":
            scaling = "parseval"
            exp = 0.5
            agg = "sum"

        # Define bin_width if octave spacing
        if octave_bins is not None:
            bin_width = 1 / slice_width

        # Loop through and compute PSD
        psd_df = pd.DataFrame()
        for i in indexes:
            window_start = max(0, i - num)
            window_end = min(length, i + num)
            with warnings.catch_warnings():
                if disable_warnings:
                    warnings.filterwarnings('ignore', '.*greater than.*', )
                slice_psd = welch(
                    df.iloc[window_start:window_end],
                    bin_width=bin_width,
                    scaling=scaling,
                    **kwargs
                )

            if octave_bins is not None or freq_splits is not None:
                with warnings.catch_warnings():
                    if disable_warnings:
                        warnings.filterwarnings('ignore', '.*empty frequency.*')
                    if freq_splits is None:
                        slice_psd = to_octave(slice_psd, octave_bins=octave_bins, fstart=fstart, agg=agg)
                    else:
                        slice_psd = to_jagged(slice_psd, freq_splits=freq_splits, agg=agg)

            if add_resultant:
                slice_psd['Resultant'] = slice_psd.sum(axis=1)

            slice_psd = (slice_psd ** exp).reset_index().melt(id_vars=slice_psd.index.name)
            slice_psd['timestamp'] = df.index[i]
            psd_df = pd.concat([psd_df, slice_psd])

    return psd_df


def spectrogram(
        df: pd.DataFrame,
        num_slices: int = 100,
        scaling: typing.Literal[None, "density", "spectrum", "parseval", "unit", "rms"] = None,
        bin_width: float = 1.0,
        octave_bins: float = None,
        fstart: float = 1.0,
        agg: typing.Union[
            typing.Literal["mean", "sum"],
            typing.Callable[[np.ndarray, int], float],
        ] = "mean",
        freq_splits: np.array = None,
        add_resultant: bool = True,
        disable_warnings: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """
    Compute a spectrogram for a time series data set using :py:func:`scipy.signal.spectrogram()`

    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param num_slices: the number of slices to split the time series into, default is 100
    :param scaling: the scaling of the output; `"density"` & `"spectrum"`
        correspond to the same options in ``scipy.signal.welch``; `"parseval"`
        will maintain the "energy" between the input & output, s.t.
        ``welch(df, scaling="parseval").sum(axis="rows")`` is roughly equal to
        ``df.abs().pow(2).sum(axis="rows")``;
        `"unit"` will maintain the units and scale of the input data. `"rms"` will use `"parseval"` for the PSD
        calculations and set `agg` to "sum", but then take the square root at the end
    :param bin_width: the bin width or resolution in Hz for the PSD or FFT, defaults to 1
    :param octave_bins: the number of frequency bins in each octave, defaults to `None`
    :param fstart: the lowest frequency for an octave spaced output, defaults to 1
    :param agg: the method for aggregating values into bins (only used if converting to octave or jagged); `'mean'` preserves
        the PSD's area-under-the-curve, `'sum'` preserves the PSD's "energy"
    :param freq_splits: the boundaries of the frequency bins to pass to :py:func:`~endaq.calc.psd.to_jagged()`
    :param add_resultant: if `True` (default) the root sum of squares of each PSD column will
        also be computed
    :param disable_warnings: if `True` (default) it disables the warnings in helper functions
    :param kwargs: Other parameters to pass directly to :py:func:`scipy.signal.spectrogram()`
    :return: a dataframe containing all the spectrograms "melted" with columns defining the value, frequency, timeslice,
        and original column from the input dataframe

    .. seealso::

        - `SciPy Spectrogram method <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html>`_
          Documentation for the function wrapped internally.
    """

    # Apply Scaling
    if scaling == "parseval":
        kwargs["scaling"] = "density"
    elif scaling == "unit":
        kwargs["scaling"] = "density"
    elif scaling == "rms":
        kwargs["scaling"] = "density"
        agg = "sum"
    elif scaling is not None:
        kwargs["scaling"] = scaling

    # Windowing, Default to Boxcar
    if "window" not in kwargs:
        kwargs["window"] = "boxcar"

    # Determine Scipy.Spectrogram Stats
    fs = 1 / utils.sample_spacing(df)
    nperseg = int(fs / bin_width)
    num_f = int(len(df) / fs / bin_width)
    noverlap = nperseg - (nperseg * num_f // num_slices)

    # Loop Through and Compute Spectrogram
    spec_df = pd.DataFrame()
    if add_resultant:
        res_df = pd.DataFrame()
    for c in df.columns:
        f, t, spec = scipy.signal.spectrogram(
            x=df[c].to_numpy(),
            fs=fs,
            nperseg=nperseg,
            mode='psd',
            noverlap=noverlap,
            **kwargs
        )

        # create dataframe
        spec_df_col = pd.DataFrame(
            data=spec,
            columns=t,
            index=f
        )
        spec_df_col.index.name = 'frequency (Hz)'

        # Do octave spacing
        if octave_bins is not None or freq_splits is not None:
            with warnings.catch_warnings():
                if disable_warnings:
                    warnings.filterwarnings('ignore', '.*empty frequency.*')
                if freq_splits is None:
                    spec_df_col = to_octave(spec_df_col, octave_bins=octave_bins, fstart=fstart, agg=agg)
                else:
                    spec_df_col = to_jagged(spec_df_col, freq_splits=freq_splits, agg=agg)

        # Melt & concatenate dataframe
        spec_df_col = spec_df_col.reset_index().melt(id_vars='frequency (Hz)')
        spec_df_col.columns = ['frequency (Hz)', 'timestamp', 'value']
        spec_df_col['variable'] = c
        spec_df = pd.concat([spec_df, spec_df_col])

        # add resultant
        if add_resultant:
            if len(res_df) == 0:
                res_df = spec_df_col
                res_df['variable'] = "Resultant"
            else:
                res_df.value += spec_df_col.value

    # Add resultant
    if add_resultant:
        spec_df = pd.concat([spec_df, res_df])

    # Update timestamps to be based on input, deal with datetime
    if isinstance(df.index[0], datetime.datetime):
        spec_df.timestamp = pd.to_timedelta(spec_df.timestamp.astype(float), unit='s') + df.index[0]
    else:
        spec_df.timestamp = spec_df.timestamp.astype(float) + df.index[0]

    # Apply scales
    if scaling == "parseval":
        spec_df.value = spec_df.value * f[1]
    if scaling == "rms":
        spec_df.value = spec_df.value ** 0.5
    elif scaling == "unit":
        spec_df.value = (spec_df.value * 2 * f[1]) ** 0.5

    return spec_df
