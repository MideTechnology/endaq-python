# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
from typing import Tuple
from collections import namedtuple
import functools
import warnings

import numpy as np
import pandas as pd
import scipy.signal

from endaq.calc.stats import L2_norm
from endaq.calc import utils


def _rel_displ_transfer_func(
    omega: float, damp: float = 0.0, dt: float = 1.0
) -> scipy.signal.ltisys.TransferFunctionDiscrete:
    """
    Generate the transfer function
        H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))
    for the PDE
        z" + (2ζω)z' + (ω²)z = -y"

    .. seealso::

        `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_

        `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
        Documentation for the transfer function class used to characterize the
        relative displacement calculation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.signal.BadCoefficients)

        return scipy.signal.TransferFunction(
            [-1],
            [1, 2 * damp * omega, omega ** 2],
        ).to_discrete(dt=dt)


def rel_displ(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the relative displacement for a SDOF system.

    The "relative" displacement follows the transfer function:
        H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))
    for the PDE:
        z" + (2ζω)z' + (ω²)z = -y"

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the relative displacement z of the SDOF system

    .. seealso::

        `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_

        `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
        Documentation for the transfer function class used to characterize the
        relative displacement calculation.

        `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`_
        Documentation for the biquad function used to implement the transfer
        function.
    """
    dt = utils.sample_spacing(accel)
    tf = _rel_displ_transfer_func(omega, damp, dt)

    return accel.apply(
        functools.partial(scipy.signal.lfilter, tf.num, tf.den, axis=0),
        raw=True,
    )


def _abs_accel_transfer_func(
    omega: float, damp: float = 0.0, dt: float = 1.0
) -> scipy.signal.ltisys.TransferFunctionDiscrete:
    """
    Generate the transfer function
        H(s) = L{x"(t)}(s) / L{y"(t)}(s) = X(s)/Y(s)
    for the PDE
        x" + (2ζω)x' + (ω²)x = (2ζω)y' + (ω²)y

    .. seealso::

        `An Introduction To The Shock Response Spectrum, Tom Irvine, 9 July 2012 <http://www.vibrationdata.com/tutorials2/srs_intr.pdf>`_

        `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
        Documentation for the transfer function class used to characterize the
        relative displacement calculation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", scipy.signal.BadCoefficients)

        return scipy.signal.TransferFunction(
            [0, 2 * damp * omega, omega ** 2],
            [1, 2 * damp * omega, omega ** 2],
        ).to_discrete(dt=dt)


def abs_accel(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the absolute acceleration for a SDOF system.

    The "absolute acceleration" follows the transfer function:
        H(s) = L{x"(t)}(s) / L{y"(t)}(s) = X(s)/Y(s)
    for the PDE:
        x" + (2ζω)x' + (ω²)x = (2ζω)y' + (ω²)y

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the absolute acceleration x" of the SDOF system

    .. seealso::

        `An Introduction To The Shock Response Spectrum, Tom Irvine, 9 July 2012 <http://www.vibrationdata.com/tutorials2/srs_intr.pdf>`_

        `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
        Documentation for the transfer function class used to characterize the
        relative displacement calculation.

        `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`_
        Documentation for the biquad function used to implement the transfer
        function.
    """
    dt = utils.sample_spacing(accel)
    tf = _abs_accel_transfer_func(omega, damp, dt)

    return accel.apply(
        functools.partial(scipy.signal.lfilter, tf.num, tf.den, axis=0),
        raw=True,
    )


def shock_spectrum(
    accel: pd.DataFrame,
    freqs: np.ndarray,
    damp: float = 0.0,
    mode: typing.Literal["srs", "pvss"] = "srs",
    two_sided: bool = False,
    aggregate_axes: bool = False,
) -> pd.DataFrame:
    """
    Calculate the shock spectrum of an acceleration signal.

    :param accel: the absolute acceleration y"
    :param freqs: the natural frequencies across which to calculate the spectrum
    :param damp: the damping coefficient ζ, related to the Q-factor by ζ = 1/(2Q);
        defaults to 0
    :param mode: the type of spectrum to calculate:

        - ``'srs'`` (default) specifies the Shock Response Spectrum (SRS)
        - ``'pvss'`` specifies the Pseudo-Velocity Shock Spectrum (PVSS)
    :param two_sided: whether to return for each frequency:
        both the maximum negative and positive shocks (`True`),
        or simply the maximum absolute shock (`False`; default)
    :param aggregate_axes: whether to calculate the column-wise resultant (`True`)
        or calculate spectra along each column independently (`False`; default)
    :return: the shock spectrum

    .. seealso::

        `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_

        `An Introduction To The Shock Response Spectrum, Tom Irvine, 9 July 2012 <http://www.vibrationdata.com/tutorials2/srs_intr.pdf>`_

        `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
        Documentation for the transfer function class used to characterize the
        relative displacement calculation.

        `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`_
        Documentation for the biquad function used to implement the transfer
        function.
    """
    if two_sided and aggregate_axes:
        raise ValueError("cannot enable both options `two_sided` and `aggregate_axes`")
    freqs = np.asarray(freqs)
    if freqs.ndim != 1:
        raise ValueError("target frequencies must be in a 1D-array")
    omega = 2 * np.pi * freqs

    if mode == "srs":
        make_tf = _abs_accel_transfer_func
    elif mode == "pvss":
        make_tf = _rel_displ_transfer_func
    else:
        raise ValueError(f"invalid spectrum mode {mode:r}")

    results = np.empty(
        (2,) + freqs.shape + ((1,) if aggregate_axes else accel.shape[1:]),
        dtype=np.float64,
    )

    dt = utils.sample_spacing(accel)
    T_padding = 1 / (
        freqs.min() * np.sqrt(1 - damp ** 2)
    )  # uses lowest damped frequency
    if not two_sided:
        T_padding /= 2

    zi = np.zeros((2,) + accel.shape[1:])
    zero_padding = np.zeros((int(T_padding // dt) + 1,) + accel.shape[1:])

    for i_nd in np.ndindex(freqs.shape):
        tf = make_tf(omega[i_nd], damp, dt)
        rd, zf = scipy.signal.lfilter(tf.num, tf.den, accel.to_numpy(), zi=zi, axis=0)
        rd_padding, _ = scipy.signal.lfilter(
            tf.num, tf.den, zero_padding, zi=zf, axis=0
        )

        if aggregate_axes:
            rd = L2_norm(rd, axis=-1, keepdims=True)
            rd_padding = L2_norm(rd_padding, axis=-1, keepdims=True)

        results[(0,) + i_nd] = -np.minimum(rd.min(axis=0), rd_padding.min(axis=0))
        results[(1,) + i_nd] = np.maximum(rd.max(axis=0), rd_padding.max(axis=0))

    if mode == "pvss":
        results = results * omega[..., np.newaxis]

    if aggregate_axes or not two_sided:
        return pd.DataFrame(
            np.maximum(results[0], results[1]),
            index=pd.Series(freqs, name="frequency (Hz)"),
            columns=(["resultant"] if aggregate_axes else accel.columns),
        )

    return namedtuple("PseudoVelocityResults", "neg pos")(
        *(
            pd.DataFrame(
                r, index=pd.Series(freqs, name="frequency (Hz)"), columns=accel.columns
            )
            for r in results
        )
    )


def enveloping_half_sine(
    pvss: pd.DataFrame,
    damp: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Characterize a half-sine pulse whose PVSS envelopes the input.

    :param pvss: the PVSS to envelope
    :param damp: the damping factor used to generate the input PVSS
    :return: a tuple of amplitudes and periods, each pair of which describes a
        half-sine pulse

    .. seealso::

        `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
    """

    def amp_factor(damp):
        """
        Calculate the PVSS amplitude attenuation on a half-sine pulse from the
        damping coefficient.

        The PVSS of a half-sine pulse differs based on the damping coefficient
        used. While the high-frequency rolloff is relatively consistent, the
        flat low-frequency amplitude is attenuated at higher damping values.
        This function calculates this attenuation for a given damping
        coefficient.
        """
        # This calculates the PVSS value as ω->0. However, since it necessarily
        # computes the maximum of a function *over time*, and ω is only found
        # therein in the multiplicative factor (ωt), it is mathematically
        # equivalent to compute this maximum for any arbitrary ω>0. Thus we
        # choose ω=1 for convenience, w/o loss of generality.
        a = np.exp(1j * np.arccos(-damp))  # = -damp + 1j * np.sqrt(1 - damp**2)
        # From WolframAlpha: https://www.wolframalpha.com/input/?i=D%5BPower%5Be%2C%5C%2840%29-d+*t%5C%2841%29%5D+sin%5C%2840%29Sqrt%5B1-Power%5Bd%2C2%5D%5D*t%5C%2841%29%2Ct%5D+%3D+0&assumption=%22ListOrTimes%22+-%3E+%22Times%22&assumption=%7B%22C%22%2C+%22e%22%7D+-%3E+%7B%22NamedConstant%22%7D&assumption=%7B%22C%22%2C+%22d%22%7D+-%3E+%7B%22Variable%22%7D&assumption=%22UnitClash%22+-%3E+%7B%22d%22%2C+%7B%22Days%22%7D%7D
        t_max = (2 / np.imag(a)) * np.arctan2(np.imag(a), 1 - np.real(a))
        PVSS_max = (1 / np.imag(a)) * np.imag(np.exp(a * t_max))
        return PVSS_max

    max_pvss = pvss.max()
    max_f_pvss = pvss.mul(pvss.index, axis=0).max()

    return namedtuple("HalfSinePulseParameters", "amplitude, period")(
        amplitude=2 * np.pi * max_f_pvss,
        period=max_pvss / (4 * amp_factor(damp) * max_f_pvss),
    )
