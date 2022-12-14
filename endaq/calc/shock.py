# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
from typing import Optional
from collections import namedtuple
from dataclasses import dataclass
import functools
import warnings

import numpy as np
import pandas as pd
import scipy.signal

from endaq.calc import utils, stats


def _absolute_acceleration_coefficients(omega, Q, T):
    """
    Calculate the coefficients of the Z-domain transfer function for the
    absolute acceleration response according to ISO 18431-4.

    :param omega: the natural frequency of the system
    :param Q: the quality factor of the system
    :param T: the time step in seconds
    :return: the coefficients of the Z-domain transfer function b, a

    .. seealso::

        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    A = omega*T/(2.*Q)
    B = omega*T*np.sqrt(1. - 1./(4.*(Q**2)))

    b = (
        1. - np.exp(-A)*np.sin(B)/B,
        2.*np.exp(-A)*(np.sin(B)/B - np.cos(B)),
        np.exp(-2*A) - np.exp(-A)*np.sin(B)/B,
        )
    a = (
        1.,
        -2.*np.exp(-A)*np.cos(B),
        np.exp(-2.*A),
        )

    return b, a


def absolute_acceleration(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the absolute acceleration for a SDOF system.

    The absolute acceleration follows the transfer function:

        `H(s) = L{x"(t)}(s) / L{y"(t)}(s) = X(s)/Y(s)`

    for the PDE:

        `x" + (2ζω)x' + (ω²)x = (2ζω)y' + (ω²)y`

    :param accel: the absolute acceleration `y"`
    :param omega: the natural frequency `ω` of the SDOF system
    :param damp: the damping coefficient `ζ` of the SDOF system
    :return: the absolute acceleration `x"` of the SDOF system

    .. seealso::

        - `An Introduction To The Shock Response Spectrum, Tom Irvine, 9 July 2012 <http://www.vibrationdata.com/tutorials2/srs_intr.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    T = utils.sample_spacing(accel)
    Q = 1./(2.*damp)

    return accel.apply(
        functools.partial(
                scipy.signal.lfilter,
                *_absolute_acceleration_coefficients(omega, Q, T),
                axis=0,
                ),
        raw=True,
        )


def _relative_velocity_coefficients(omega, Q, T):
    """
    Calculate the coefficients of the Z-domain transfer function for the
    relative velocity response according to ISO 18431-4.

    :param omega: the natural frequency of the system
    :param Q: the quality factor of the system
    :param T: the time step in seconds
    :return: the coefficients of the Z-domain transfer function b, a

    .. seealso::

        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    A = omega*T/(2.*Q)
    B = omega*T*np.sqrt(1. - 1./(4.*(Q**2.)))
    C = np.exp(-A)*np.sin(B)/np.sqrt(4.*(Q**2.) - 1.)
    D = T*(omega**2.)

    b = (
        (-1. + np.exp(-A)*np.cos(B) + C)/D,
        (1. - np.exp(-2.*A) - 2.*C)/D,
        (np.exp(-2.*A) - np.exp(-A)*np.cos(B) + C)/D,
        )
    a = (
        1.,
        -2.*np.exp(-A)*np.cos(B),
        np.exp(-2.*A),
        )

    return b, a


def relative_velocity(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the relative velocity for a SDOF system.

    The relative velocity follows the transfer function:

        `H(s) = L{z'(t)}(s) / L{y"(t)}(s) = (1/s)(Z(s)/Y(s))`

    for the PDE:

        `z" + (2ζω)z' + (ω²)z = -y"`

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the relative velocity z' of the SDOF system

    .. seealso::

        - `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    T = utils.sample_spacing(accel)
    Q = 1./(2.*damp)

    return accel.apply(
        functools.partial(
                scipy.signal.lfilter,
                *_relative_velocity_coefficients(omega, Q, T),
                axis=0,
                ),
        raw=True,
        )


def _relative_displacement_coefficients(omega, Q, T):
    """
    Calculate the coefficients of the Z-domain transfer function for the
    relative displacement response according to ISO 18431-4.

    :param omega: the natural frequency of the system
    :param Q: the quality factor of the system
    :param T: the time step in seconds
    :return: the coefficients of the Z-domain transfer function b, a

    .. seealso::

        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    A = omega*T/(2.*Q)
    B = omega*T*np.sqrt(1. - 1./(4.*(Q**2.)))
    C = T*(omega**3.)
    q = (1./(2.*(Q**2.)) - 1.)/np.sqrt(1. - 1./(4.*(Q**2.)))

    b = (
        ((1. - np.exp(-A)*np.cos(B))/Q - q*np.exp(-A)*np.sin(B) - omega*T)/C,
        (2.*np.exp(-A)*np.cos(B)*omega*T -
         (1. - np.exp(-2.*A))/Q +
         2*q*np.exp(-A)*np.sin(B))/C,
        (-np.exp(-2.*A)*(omega*T + 1./Q) +
         np.exp(-A)*np.cos(B)/Q -
         q*np.exp(-A)*np.sin(B))/C,
        )
    a = (
        1.,
        -2.*np.exp(-A)*np.cos(B),
        np.exp(-2.*A),
        )

    return b, a


def relative_displacement(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the relative displacement for a SDOF system.

    The relative displacement follows the transfer function:

        `H(s) = L{z(t)}(s) / L{y"(t)}(s) = (1/s²)(Z(s)/Y(s))`

    for the PDE:

        `z" + (2ζω)z' + (ω²)z = -y"`

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the relative displacement z of the SDOF system

    .. seealso::

        - `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    T = utils.sample_spacing(accel)
    Q = 1./(2.*damp)

    return accel.apply(
        functools.partial(
                scipy.signal.lfilter,
                *_relative_displacement_coefficients(omega, Q, T),
                axis=0,
                ),
        raw=True,
        )


def _pseudo_velocity_coefficients(omega, Q, T):
    """
    Calculate the coefficients of the Z-domain transfer function for the
    pseudo-velocity response according to ISO 18431-4.

    :param omega: the natural frequency of the system
    :param Q: the quality factor of the system
    :param T: the time step in seconds
    :return: the coefficients of the Z-domain transfer function b, a

    .. seealso::

        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    A = omega*T/(2.*Q)
    B = omega*T*np.sqrt(1. - 1./(4.*(Q**2)))
    C = T*(omega**2)
    q = (1./(2.*(Q**2.)) - 1.)/np.sqrt(1. - 1./(4.*(Q**2.)))

    b = (
        ((1. - np.exp(-A)*np.cos(B))/Q - q*np.exp(-A)*np.sin(B) - omega*T)/C,
        (2.*np.exp(-A)*np.cos(B)*omega*T - (1. - np.exp(-2.*A))/Q + 2.*q*np.exp(-A)*np.sin(B))/C,
        (-np.exp(-2.*A)*(omega*T + 1./Q) + np.exp(-A)*np.cos(B)/Q - q*np.exp(-A)*np.sin(B))/C,
        )
    a = (
        1.,
        -2.*np.exp(-A)*np.cos(B),
        np.exp(-2.*A),
        )

    return b, a


def pseudo_velocity(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the pseudo-velocity for a SDOF system.

    The pseudo-velocity follows the transfer function:

        `H(s) = L{ωz(t)}(s) / L{y"(t)}(s) = (ω/s²)(Z(s)/Y(s))`

    for the PDE:

        `z" + (2ζω)z' + (ω²)z = -y"`

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the pseudo-velocity of the SDOF system

    .. seealso::

        - `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    T = utils.sample_spacing(accel)
    Q = 1./(2.*damp)

    return accel.apply(
        functools.partial(
                scipy.signal.lfilter,
                *_pseudo_velocity_coefficients(omega, Q, T),
                axis=0,
                ),
        raw=True,
        )


def _relative_displacement_static_coefficients(omega, Q, T):
    """
    Calculate the coefficients of the Z-domain transfer function for the
    relative displacement response expressed as equivalent static acceleration
    according to ISO 18431-4.

    :param omega: the natural frequency of the system
    :param Q: the quality factor of the system
    :param T: the time step in seconds
    :return: the coefficients of the Z-domain transfer function b, a

    .. seealso::

        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    A = omega*T/(2.*Q)
    B = omega*T*np.sqrt(1. - 1/(4.*(Q**2.)))
    C = (T*omega)
    q = (1./(2.*(Q**2.)) - 1.)/(np.sqrt(1. - 1./(4.*(Q**2.))))

    b = (
        ((1 - np.exp(-A)*np.cos(B))/Q - q*np.exp(-A)*np.sin(B) - omega*T)/C,
        (2*np.exp(-A)*np.cos(B)*omega*T - (1 - np.exp(-2.*A))/Q + 2.*q*np.exp(-A)*np.sin(B))/C,
        (-np.exp(-2.*A)*(omega*T + 1./Q) + np.exp(-A)*np.cos(B)/Q - q*np.exp(-A)*np.sin(B))/C,
        )
    a = (
        1.,
        -2.*np.exp(-A)*np.cos(B),
        np.exp(-2.*A),
        )

    return b, a


def relative_displacement_static(accel: pd.DataFrame, omega: float, damp: float = 0.0) -> pd.DataFrame:
    """
    Calculate the relative displacement expressed as equivalent static
    acceleration for a SDOF system.

    The relative displacement as static acceleration follows the transfer
    function:

        `H(s) = L{ω²z(t)}(s) / L{y"(t)}(s) = (ω²/s²)(Z(s)/Y(s))`

    for the PDE:

        `z" + (2ζω)z' + (ω²)z = -y"`

    :param accel: the absolute acceleration y"
    :param omega: the natural frequency ω of the SDOF system
    :param damp: the damping coefficient ζ of the SDOF system
    :return: the relative displacement of the SDOF system expressed as
        equivalent static acceleration

    .. seealso::

        - `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
        - `ISO 18431-4 Mechanical vibration and shock — Signal processing — Part 4: Shock-response spectrum analysis`
          Explicit implementations of digital filter coefficients for shock spectra.
    """
    T = utils.sample_spacing(accel)
    Q = 1./(2.*damp)

    return accel.apply(
        functools.partial(
                scipy.signal.lfilter,
                *_relative_displacement_static_coefficients(omega, Q, T),
                axis=0,
                ),
        raw=True,
        )


def shock_spectrum(
    accel: pd.DataFrame,
    freqs: np.ndarray = None,
    init_freq: float = 0.5,
    bins_per_octave: float = 12.0,
    damp: float = 0.05,
    mode: typing.Literal["srs", "pvss"] = "srs",
    max_time: float = 2.0,
    peak_threshold: float = 0.1,
    two_sided: bool = False,
    aggregate_axes: bool = False,
) -> pd.DataFrame:
    """
    Calculate the shock spectrum of an acceleration signal. Note this defaults to first find peak events, then compute
    the spectrum on those peaks to speed up processing time.
    
    :param accel: the absolute acceleration `y"`
    :param freqs: the natural frequencies across which to calculate the spectrum,
        if `None` (the default) it uses `init_freq` and `bins_per_octave` to define them
    :param init_freq: the initial frequency in the sequence; if `None`,
        use the frequency corresponding to the data's duration, default is 0.5 Hz
    :param bins_per_octave: the number of frequencies per octave, default is 12    
    :param damp: the damping coefficient `ζ`, related to the Q-factor by
        `ζ = 1/(2Q)`; defaults to 0.05
    :param mode: the type of spectrum to calculate:

        *  `'srs'` (default) specifies the Shock Response Spectrum (SRS)
        *  `'pvss'` specifies the Pseudo-Velocity Shock Spectrum (PVSS)
    :param max_time: the maximum duration in seconds to compute the shock spectrum for, if the time duration is greater
        than :py:func:`~endaq.calc.stats.find_peaks()` is used to find peak locations, then the shock spectrums at each
        peak is calculated with :py:func:`~endaq.calc.shock.rolling_shock_spectrum()` with `max_time` defining the
        `slice_width`. Set `max_time` to `None` to force the function to not do the peak finding.
    :param peak_threshold: if the duration is greater than `max_time` all peaks that are greater than `peak_threshold`
        will be calculated, and the aggregate max per frequency will be reported
    :param two_sided: whether to return for each frequency:
        both the maximum negative and positive shocks (`True`),
        or simply the maximum absolute shock (`False`; default)
    :param aggregate_axes: whether to calculate the column-wise resultant (`True`)
        or calculate spectra along each column independently (`False`; default)
    :return: the shock spectrum
    .. seealso::
        - `Pseudo Velocity Shock Spectrum Rules For Analysis Of Mechanical Shock, Howard A. Gaberson <https://info.endaq.com/hubfs/pvsrs_rules.pdf>`_
        - `An Introduction To The Shock Response Spectrum, Tom Irvine, 9 July 2012 <http://www.vibrationdata.com/tutorials2/srs_intr.pdf>`_
        - `SciPy transfer functions <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html>`_
          Documentation for the transfer function class used to characterize the
          relative displacement calculation.
        - `SciPy biquad filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html>`__
          Documentation for the biquad function used to implement the transfer
          function.
    """
    if freqs is None:
        freqs = utils.logfreqs(accel, init_freq=init_freq, bins_per_octave=bins_per_octave)
    if two_sided and aggregate_axes:
        raise ValueError("cannot enable both options `two_sided` and `aggregate_axes`")
    freqs = np.asarray(freqs)
    if freqs.ndim != 1:
        raise ValueError("target frequencies must be in a 1D-array")

    # Check for total time, if too long compute only for peaks
    dt = utils.sample_spacing(accel)
    if max_time is None:
        max_time = dt * len(accel) * 2
    if dt * len(accel) > max_time:
        rolling_srs = rolling_shock_spectrum(
            accel,
            damp=damp,
            mode=mode,
            add_resultant=aggregate_axes,
            freqs=freqs,
            indexes=stats.find_peaks(accel, time_distance=max_time, threshold_multiplier=peak_threshold),
            slice_width=max_time
        )
        var_name = 'variable'
        if 'axis' in rolling_srs.columns:
            var_name = 'axis'
        srs = pd.DataFrame(
            index=pd.Series(rolling_srs['frequency (Hz)'].unique(), name='frequency (Hz)'),
            columns=rolling_srs[var_name].unique()
        )
        for var in srs.columns:
            pivot = rolling_srs[rolling_srs[var_name] == var].copy()
            pivot = pivot.pivot_table(
                columns='timestamp',
                index='frequency (Hz)',
                values='value'
            )
            srs[var] = pivot.max(axis=1)
        srs.columns.name = accel.columns.name
        return srs

    omega = 2 * np.pi * freqs

    if mode == "srs":
        make_coeffs = _absolute_acceleration_coefficients
    elif mode == "pvss":
        make_coeffs = _pseudo_velocity_coefficients
    else:
        raise ValueError(f"invalid spectrum mode {mode:r}")

    results = np.empty(
        (2,) + freqs.shape + ((1,) if aggregate_axes else accel.shape[1:]),
        dtype=np.float64,
    )

    T_padding = 1 / (
        freqs.min() * np.sqrt(1 - damp ** 2)
    )  # uses lowest damped frequency
    if not two_sided:
        T_padding /= 2

    zi = np.zeros((2,) + accel.shape[1:])
    zero_padding = np.zeros((int(T_padding // dt) + 1,) + accel.shape[1:])

    Q = 1./(2.*damp)

    for i_nd in np.ndindex(freqs.shape[0]):
        rd, zf = scipy.signal.lfilter(
                *make_coeffs(omega[i_nd], Q, dt),
                accel.to_numpy(),
                zi=zi,
                axis=0,
                )
        rd_padding, _ = scipy.signal.lfilter(
            *make_coeffs(omega[i_nd], Q, dt), zero_padding, zi=zf, axis=0
        )

        if aggregate_axes:
            rd = stats.L2_norm(rd, axis=-1, keepdims=True)
            rd_padding = stats.L2_norm(rd_padding, axis=-1, keepdims=True)

        results[(0,) + i_nd] = -np.minimum(rd.min(axis=0), rd_padding.min(axis=0))
        results[(1,) + i_nd] = np.maximum(rd.max(axis=0), rd_padding.max(axis=0))

    if aggregate_axes or not two_sided:
        return pd.DataFrame(
            np.maximum(results[0], results[1]),
            index=pd.Series(freqs, name="frequency (Hz)"),
            columns=(["Resultant"] if aggregate_axes else accel.columns),
        )

    return namedtuple("PseudoVelocityResults", "neg pos")(
        *(
            pd.DataFrame(
                r, index=pd.Series(freqs, name="frequency (Hz)"), columns=accel.columns
            )
            for r in results
        )
    )


def rolling_shock_spectrum(
        df: pd.DataFrame,
        damp: float = 0.05,
        mode: typing.Literal["srs", "pvss"] = "srs",
        add_resultant: bool = True,
        freqs: np.ndarray = None,
        init_freq: float = 0.5,
        bins_per_octave: float = 12.0,
        num_slices: int = 100,
        indexes: np.array = None,
        index_values: np.array = None,
        slice_width: float = None,
        disable_warnings: bool = True,
) -> pd.DataFrame:
    """
    Compute Shock Response Spectrums for defined slices of a time series data set using :py:func:`~endaq.calc.shock.shock_spectrum()`
    
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param damp: the damping coefficient `ζ`, related to the Q-factor by
        `ζ = 1/(2Q)`; defaults to 0.05
    :param mode: the type of spectrum to calculate:

        *  `'srs'` (default) specifies the Shock Response Spectrum (SRS)
        *  `'pvss'` specifies the Pseudo-Velocity Shock Spectrum (PVSS)
    :param add_resultant: if `True` (default) the column-wise resultant will
        also be computed and returned with the spectra per-column
    :param freqs: the natural frequencies across which to calculate the spectrum,
        if `None` (the default) it uses `init_freq` and `bins_per_octave` to define them
    :param init_freq: the initial frequency in the sequence; if `None`,
        use the frequency corresponding to the data's duration, default is 0.5 Hz
    :param bins_per_octave: the number of frequencies per octave, default is 12    
    :param num_slices: the number of slices to split the time series into, default is 100,
        this is ignored if `indexes` is defined
    :param indexes: the center index locations (not value) of each slice to compute the shock spectrum
    :param index_values: the index values of each peak event to quantify (slower but more intuitive than using `indexes`)
    :param slice_width: the time in seconds to center about each slice index,
        if none is provided it will calculate one based upon the number of slices
    :param disable_warnings: if `True` (default) it disables the warnings on the initial frequency
    :return: a dataframe containing all the shock spectrums, stacked on each other
    
    See example use cases and syntax at :py:func:`~endaq.plot.spectrum_over_time()`
    which visualizes the output of this function in Heatmaps, Waterfall plots, 
    Surface plots, and Animations

    """
    if freqs is None:
        freqs = utils.logfreqs(df, init_freq=init_freq, bins_per_octave=bins_per_octave)

    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df,
        indexes=indexes,
        index_values=index_values,
        num_slices=num_slices,
        slice_width=slice_width
    )

    # Loop through and compute shock spectrum
    srs = pd.DataFrame()
    for i in indexes:
        window_start = max(0, i - num)
        window_end = min(length, i + num)
        with warnings.catch_warnings():
            if disable_warnings:
                warnings.filterwarnings('ignore', '.*too short*', )
            slice_srs = shock_spectrum(
                df.iloc[window_start:window_end],
                mode=mode,
                damp=damp,
                freqs=freqs,
            )
        if add_resultant:
            with warnings.catch_warnings():
                if disable_warnings:
                    warnings.filterwarnings('ignore', '.*too short*', )
                slice_srs['Resultant'] = shock_spectrum(
                    df.iloc[window_start:window_end],
                    mode=mode,
                    damp=damp,
                    freqs=freqs,
                    aggregate_axes=True
                )['Resultant']

        slice_srs = slice_srs.reset_index().melt(id_vars=slice_srs.index.name)
        slice_srs['timestamp'] = df.index[i]
        srs = pd.concat([srs, slice_srs])

    return srs


@dataclass
class HalfSineWavePulse:
    """
    The output data type for :py:func:`enveloping_half_sine`.

    The significant data members are `amplitude` and `duration`, which can
    simply be unpacked as if from a plain tuple:

    .. testsetup::

        import pandas as pd
        df_pvss = pd.DataFrame([1, 1], index=[200, 400])

        from endaq.calc.shock import enveloping_half_sine

    .. testcode::

        ampl, T = enveloping_half_sine(df_pvss)

    However, users can also elect to use the other methods of this class to
    generate other kinds of outputs.

    .. note:: This class is not intended to be instantiated manually.
    """

    amplitude: pd.Series
    duration: pd.Series

    def __iter__(self):
        return iter((self.amplitude, self.duration))

    def to_time_series(
        self,
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        dt: Optional[float] = None,
        tpulse: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a time-series of the half-sine pulse.

        :param tstart: the starting time of the resulting waveform; if `None`
            (default), the range starts at `tpulse`
        :param tstop: the ending time of the resulting waveform; if `None`
            (default), the range ends at `tpulse + duration`
        :param dt: the sampling period of the resulting waveform; defaults to
            1/20th of the pulse duration
        :param tpulse: the starting time of the pulse within the resulting
            waveform; if `None` (default), the pulse starts at either:

            - ``tstart``, if provided
            - ``tstop - self.duration.max()``, if `tstop` is provided
            - ``0.0`` otherwise
        :return: a time-series of the half-sine pulse
        """
        if dt is None:
            dt = self.duration.min() / 20
        if dt > self.duration.min() / 8:
            warnings.warn(
                f"the sampling period {dt} is large relative to the pulse duration"
                f" {self.duration.min()}; the waveform may not accurately represent"
                f" the half-sine pulse's shock intensity"
            )

        default_start = 0.0
        if tstop is not None:
            default_start = tstop - self.duration.max()

        if tpulse is None and tstart is None:
            tpulse = tstart = default_start
        elif tpulse is None:
            tpulse = tstart
        elif tstart is None:
            tstart = tpulse

        if tstop is None:
            tstop = tpulse + self.duration.max()

        if not (tstart <= tpulse <= tstop - self.duration.max()):
            warnings.warn(
                "half-sine pulse extends beyond the bounds of the time series"
            )

        t = np.arange(tstart, tstop, dt)

        data = np.zeros((len(t), len(self.amplitude)), dtype=float)
        t_data, ampl_data, T_data = np.broadcast_arrays(
            t[..., None], self.amplitude.to_numpy(), self.duration.to_numpy()
        )
        t_mask = np.nonzero((t_data >= tpulse) & (t_data < tpulse + T_data))
        data[t_mask] = ampl_data[t_mask] * np.sin(
            np.pi * t_data[t_mask] / T_data[t_mask]
        )

        return pd.DataFrame(
            data,
            index=pd.Series(t, name="timestamp"),
            columns=self.amplitude.index,
        )

    # def widened_duration(self, new_duration: float):
    #    pass

    # def pseudo_velocity(self):
    #    pass


def enveloping_half_sine(
    pvss: pd.DataFrame,
    damp: float = 0.0,
) -> HalfSineWavePulse:
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

    return HalfSineWavePulse(
        amplitude=2 * np.pi * max_f_pvss,
        duration=max_pvss / (4 * amp_factor(damp) * max_f_pvss),
    )
