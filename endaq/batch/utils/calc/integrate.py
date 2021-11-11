import scipy.integrate
import scipy.signal

from . import filters


def _integrate(array, dt, axis=-1):
    """Integrate data over an axis."""
    result = scipy.integrate.cumtrapz(array, dx=dt, initial=0, axis=axis)
    # In lieu of explicit initial offset, set integration bias to remove mean
    # -> avoids trend artifacts after successive integrations
    result = result - result.mean(axis=axis, keepdims=True)

    return result


def iter_integrals(array, dt, axis=-1, highpass_cutoff=1.0):
    """Iterate over conditioned integrals of the given original data."""
    array = filters.highpass(
        array, fs=1 / dt, half_order=3, cutoff=highpass_cutoff, axis=axis
    )
    while True:
        array.setflags(write=False)  # should NOT mutate shared data
        yield array
        array.setflags(write=True)  # array will be replaced below -> now ok to edit
        array = _integrate(array, dt, axis=axis)
