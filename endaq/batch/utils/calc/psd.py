import numbers
import warnings

import numpy as np
import scipy.signal


def welch(array, nfft=1024, dt=None, axis=-1):
    """
    Calculate a periodogram using a modified welch's method.

    The modified method is designed to maintain equivalence between the input
    and the result as dictated by Parseval's Theorem. In order to achieve this,
    this method is modified to:
    - generate a spectrogram using a flat boxcar window
    - use zero overlap in the STFT
    - pad the signal with zeros when windows extend over the edge of the array
    - sum each frequency bin across time (instead of averaging)
    """
    warnings.warn(
        "this method is deprecated; use instead `scipy.signal.welch`",
        DeprecationWarning,
    )

    if not isinstance(nfft, numbers.Integral):
        raise TypeError
    if dt is not None:
        if not isinstance(dt, numbers.Real):
            raise TypeError
        if dt <= 0:
            raise ValueError

    array = np.moveaxis(array, axis, -1)
    _f, _t, a_stft = scipy.signal.stft(
        array,
        fs=1,
        window="boxcar",
        nperseg=nfft,
        noverlap=0,
        nfft=nfft,
        boundary="zeros",
        padded=True,
        axis=-1,
    )
    assert a_stft.shape[-2] == nfft // 2 + 1
    assert a_stft.shape[-1] == -((array.shape[-1] - 1) // -nfft) + 1

    a_psd = np.sum(np.abs(a_stft) ** 2, axis=-1)
    a_psd[..., 1 : (nfft - 1) // 2 + 1] *= 2
    a_psd *= nfft ** 2 / array.shape[-1]

    a_psd = np.moveaxis(a_psd, -1, axis)

    if dt is None:
        return a_psd
    else:
        freqs = np.fft.rfftfreq(nfft, d=dt)
        return freqs, dt * a_psd


def rms(array, fs, dn=0, min_freq=0, max_freq=np.inf, nfft=1024, axis=-1):
    """
    Multi-axis RMS of nth derivative.

    :param array: the source data
    :param fs: the sampling frequency
    :param dn: the derivative number (e.g. 1 = first derivative, 2 = second,
        -1 = first anti-derivative)
    :param highpass_freq: the frequency up to which all frequency content is
        set to zero (inclusive; default to 0)
    """
    array = np.moveaxis(array, axis, -1)
    if fs <= 0:
        raise ValueError
    if not isinstance(dn, numbers.Integral):
        raise TypeError

    a_psd = welch(array, nfft=nfft, axis=-1)
    a_freqs = np.fft.rfftfreq(nfft, d=1 / fs)

    v_psd = a_psd
    # Bandpass filtering
    i_min_freq, i_max_freq = np.searchsorted(
        a_freqs, [min_freq, max_freq], side="right"
    )
    v_psd[..., :i_min_freq] = 0
    v_psd[..., i_max_freq:] = 0
    # Time-domain derivation/integration
    v_psd[..., i_min_freq:i_max_freq] *= a_freqs[i_min_freq:i_max_freq] ** (2 * dn)

    return np.sqrt(np.sum(v_psd, axis=-1) / nfft)


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


def differentiate(f, psd, n=1):
    """Perform time-domain differentiation on periodogram data."""
    # Involves a division by zero for n<0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        factor = (2 * np.pi * f) ** (2 * n)  # divide by zero
    if n < 0:
        factor[f == 0] = 0

    return f, psd * factor


def to_jagged(f, psd, freq_splits, axis=-1, mode="sum"):
    """
    Calculate a periodogram over non-uniformly spaced frequency bins.

    :param f, psd: the returned values from `scipy.signal.welch`
    :param freq_splits: the boundaries of the frequency bins; must be strictly
        increasing
    :param axis: same as the axis parameter provided to `scipy.signal.welch`
    :param mode: the method for aggregating values into bins; 'mean' preserves
        the PSD's area-under-the-curve, 'sum' preserves the PSD's "energy"
    """
    if not np.all(np.diff(freq_splits, prepend=0) > 0):
        raise ValueError

    # Check that PSD samples do not skip any frequency bins
    spacing_test = np.diff(np.searchsorted(freq_splits, f))
    if np.any(spacing_test > 1):
        warnings.warn(
            "empty frequency bins in re-binned PSD; "
            "original PSD's frequency spacing is too coarse",
            RuntimeWarning,
        )

    psd_jagged = _np_histogram_nd(f, bins=freq_splits, weights=psd)
    if mode == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # x/0

            psd_jagged = np.nan_to_num(  # <- fix divisions by zero
                psd_jagged / np.histogram(f, bins=freq_splits)[0]
            )

    f = (freq_splits[1:] + freq_splits[:-1]) / 2
    return f, psd_jagged


def to_octave(f, psd, fstart=1, octave_bins=12, **kwargs):
    """Calculate a periodogram over log-spaced frequency bins."""
    max_f = f.max()

    octave_step = 1 / octave_bins
    center_freqs = 2 ** np.arange(
        np.log2(fstart),
        np.log2(max_f) - octave_step / 2,
        octave_step,
    )
    freq_splits = 2 ** np.arange(
        np.log2(fstart) - octave_step / 2,
        np.log2(max_f),
        octave_step,
    )
    assert len(center_freqs) + 1 == len(freq_splits)

    return center_freqs, to_jagged(f, psd, freq_splits=freq_splits, **kwargs)[1]
