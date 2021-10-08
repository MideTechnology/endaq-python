import numpy as np
import scipy.signal


def highpass(array, fs, cutoff=1.0, half_order=3, axis=-1):
    """Apply a highpass filter to an array."""
    array = np.moveaxis(array, axis, -1)

    sos_coeffs = scipy.signal.butter(
        N=half_order,
        Wn=cutoff,
        btype="highpass",
        fs=fs,
        output="sos",
    )

    # vvv
    init_state = scipy.signal.sosfilt_zi(sos_coeffs)
    for _ in range(2):
        init_fwd = init_state * array[(Ellipsis, 0) + ((None,) * init_state.ndim)]
        init_fwd = np.moveaxis(init_fwd, array.ndim - 1, 0)
        array, _zo = scipy.signal.sosfilt(sos_coeffs, array, axis=-1, zi=init_fwd)
        array = array[..., ::-1]
    # ^^^ could alternatively do this (not as good though?):
    # array = scipy.signal.sosfiltfilt(sos_coeffs, array, axis=-1)

    return np.moveaxis(array, -1, axis)
