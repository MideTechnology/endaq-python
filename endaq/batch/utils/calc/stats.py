import numpy as np
import scipy.ndimage


def L2_norm(data, axis=None, keepdims=False):
    """Compute the L2 norm (a.k.a. the Euclidean Norm)."""
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis, keepdims=keepdims))


def max_abs(array, axis=None, keepdims=False):
    """
    Compute the maximum of the absolute value of an array.

    This function should be equivalent to, but generally use less memory than
    `np.amax(np.abs(array))`.

    Specifically, it generates the absolute-value maximum from `np.amax(array)`
    and `-np.amin(array)`. Thus instead of allocating space for the intermediate
    array `np.abs(array)`, it allocates for the axis-collapsed smaller arrays
    `np.amax(array)` & `np.amin(array)`.

    Note - this method does not work on complex-valued arrays.
    """
    # Forbid complex-valued data
    if np.iscomplexobj(array):
        raise ValueError("`max_abs` does not accept complex arrays")

    return np.maximum(
        np.amax(array, initial=-np.inf, axis=axis, keepdims=keepdims),
        -np.amin(array, initial=np.inf, axis=axis, keepdims=keepdims),
    )


def rms(data, axis=None, keepdims=False):
    """Calculate the root-mean-square (RMS) along a given axis."""
    return np.sqrt(np.mean(np.abs(data) ** 2, axis=axis, keepdims=keepdims))


def rolling_rms(array, nperseg=256, axis=-1):
    """Calculate a rolling RMS along a given axis."""
    sq = array ** 2
    window = np.ones(nperseg, dtype=array.dtype) / nperseg
    mean_sq = scipy.ndimage.convolve1d(sq, window, axis=axis, mode="mirror")
    return np.sqrt(mean_sq)
