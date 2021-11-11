import numbers
import warnings

import numpy as np


class NegativeOverlapWarning(RuntimeWarning):
    """A warning class for segment views with missing data."""


def segment_view(
    array: np.ndarray,
    nperseg: numbers.Integral,
    step: numbers.Integral,
    axis=-1,
):
    """
    Create a new view of a numpy array, splitting an axis into segments.

    :param array: the input array
    :param nperseg: the window size over which to segment the array
    :param step: how far to move the window between consecutive segments;
        related to `noverlap` parameter in analogous functions by
        `step == nperseg - noverlap`
    :param axis: the axis on which to segment the data
    """
    if not isinstance(nperseg, numbers.Integral):
        raise TypeError(f"non-int nperseg type {type(nperseg)}")
    if nperseg <= 0:
        raise ValueError(f"invalid non-positive nperseg {nperseg}")

    if not isinstance(step, numbers.Integral):
        raise TypeError(f"non-int step type {type(step)}")
    if step <= 0:
        raise ValueError(f"invalid non-positive step size {step}")

    axis = axis % array.ndim  # index axis from front
    array = np.moveaxis(array, axis, -1)

    # Avoid overflow when converting to C-long in strides
    nperseg = min(nperseg, array.shape[-1] + 1)
    step = min(step, array.shape[-1] + 1)

    segment_count = 1 + (array.shape[-1] - nperseg) // step
    if segment_count > 1 and nperseg < step:
        warnings.warn(
            "negative overlap on segments; some data not present in view",
            NegativeOverlapWarning,
        )

    view = np.lib.stride_tricks.as_strided(
        array,
        shape=array.shape[:-1] + (segment_count, nperseg),
        strides=array.strides[:-1] + (step * array.strides[-1], array.strides[-1]),
        writeable=False,  # necessary in case of overlap
    )

    i_last_segment = segment_count * step
    assert array.shape[-1] - nperseg < i_last_segment

    view = np.moveaxis(view, [-2, -1], [axis, axis + 1])
    excess = np.moveaxis(array[..., i_last_segment:], -1, axis)
    return view, excess
