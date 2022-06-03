# -*- coding: utf-8 -*-

from unittest import mock
import warnings

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import numpy.testing as npt
import pandas as pd
import sympy as sp

from endaq.calc import shock


wn, fn, wi, fi, Q, d, T = sp.symbols('ωₙ, fₙ, ωᵢ, fᵢ, Q, ζ, T', real=True)
s = sp.Symbol('s', complex=True)

DAMP_RANGE = (1e-2, 1.)


def laplace(b, a, freqs, subs):

    # first substitution
    b = b.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)
    a = a.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)

    b_nums = sp.lambdify(fi, b)(freqs)
    a_nums = sp.lambdify(fi, a)(freqs)

    mag = abs(b_nums)/abs(a_nums)
    phase = np.angle(b_nums) - np.angle(a_nums)

    return mag, phase


def laplace_amplitude(b, a, freqs, subs):

    # first substitution
    b = b.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)
    a = a.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)

    mag = sp.lambdify(fi, abs(b)/abs(a))

    return mag(freqs)


def laplace_phase(b, a, freqs, subs):

    # first substitution
    b = b.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)
    a = a.subs({wn: 2*sp.pi*fn, Q: 1/(2*d), s: sp.I*2*sp.pi*fi}).subs(subs)

    phase = sp.lambdify(fi, sp.atan2(*b.as_real_imag()[::-1]) - sp.atan2(*a.as_real_imag()[::-1]))

    return phase(freqs)


def z_amplitude(b, a, freqs, dt):

    z = sp.exp(-s*T)

    b = sum([x*z**i for i, x in enumerate(b)])
    a = sum([x*z**i for i, x in enumerate(a)])

    # first substitution
    b = b.subs({s: sp.I*2*sp.pi*fi, T: dt})
    a = a.subs({s: sp.I*2*sp.pi*fi, T: dt})

    mag = sp.lambdify(fi, abs(b)/abs(a))

    return abs(mag(freqs))


def z_phase(b, a, freqs, dt):

    z = sp.exp(-s*T)

    b = sum([x*z**i for i, x in enumerate(b)])
    a = sum([x*z**i for i, x in enumerate(a)])

    # first substitution
    b = b.subs({s: sp.I*2*sp.pi*fi, T: dt})
    a = a.subs({s: sp.I*2*sp.pi*fi, T: dt})

    phase = sp.lambdify(fi, sp.atan2(*b.as_real_imag()[::-1]) - sp.atan2(*a.as_real_imag()[::-1]))

    return phase(freqs)


@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_rel_displ_amp(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -1
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q
    With the amplitude response of:
                  |a₂(ωᵢ*j)|
    |G(ωᵢ*j)|  = ------------
                  |a₁(ωᵢ*j)|

    """
    dt = 1e-4
    omega = 2 * np.pi * freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_amplitude(sp.sympify(-1), s**2 + wn*s/Q + wn**2, freqs, {fn: freq, d: damp})
    za = z_amplitude(*shock._relative_displacement_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)

@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_rel_displ_phase(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -1
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q

    With the phase response of:
    ∠G(ωᵢ*j) = ∠a₂(ωᵢ*j) - ∠a₁(ωᵢ*j)
    """
    dt = 1e-4
    omega = 2*np.pi*freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_phase(sp.sympify(-1), s**2 + wn*s/Q + wn**2, freqs, {fn:freq, d:damp})
    za = z_phase(*shock._relative_displacement_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)


@hyp.given(
        freq=hyp_st.floats(12.5, 1000),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_rel_velocity_amp(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -s
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q
    With the amplitude response of:
                  |a₂(ωᵢ*j)|
    |G(ωᵢ*j)|  = ------------
                  |a₁(ωᵢ*j)|

    """
    dt = 1e-4
    omega = 2 * np.pi * freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_amplitude(-s, s**2 + wn*s/Q + wn**2, freqs, {fn: freq, d: damp})
    za = z_amplitude(*shock._relative_velocity_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)

@hyp.given(
        freq=hyp_st.floats(12.5, 1000),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_rel_velocity_phase(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -s
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q

    With the phase response of:
    ∠G(ωᵢ*j) = ∠a₂(ωᵢ*j) - ∠a₁(ωᵢ*j)
    """
    dt = 1e-4
    omega = 2*np.pi*freq

    freqs = np.concatenate([np.geomspace(1e-1, freq*0.99), [], np.geomspace(freq*1.01, 2e3)])

    la = laplace_phase(-s, s**2 + wn*s/Q + wn**2, freqs, {fn:freq, d:damp})
    za = z_phase(*shock._relative_velocity_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)

@hyp.given(
        freq=hyp_st.floats(12.5, 1000),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_abs_accel_amp(freq, damp):
    """
    Laplace domain transfer function:
                       ωₙ*s
                      ----- + ωₙ²
           a₂(s)        Q
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q
    With the amplitude response of:
                  |a₂(ωᵢ*j)|
    |G(ωᵢ*j)|  = ------------
                  |a₁(ωᵢ*j)|

    """
    dt = 1e-4
    omega = 2 * np.pi * freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_amplitude(wn*s/Q + wn**2, s**2 + wn*s/Q + wn**2, freqs, {fn: freq, d: damp})
    za = z_amplitude(*shock._absolute_acceleration_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)


@hyp.given(
        freq=hyp_st.floats(12.5, 1000),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_abs_accel_phase(freq, damp):
    """
                       ωₙ*s
                      ----- + ωₙ²
           a₂(s)        Q
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q

    With the phase response of:
    ∠G(ωᵢ*j) = ∠a₂(ωᵢ*j) - ∠a₁(ωᵢ*j)
    """
    dt = 1e-4
    omega = 2*np.pi*freq

    freqs = np.concatenate([np.geomspace(1e-1, freq*0.99), [], np.geomspace(freq*1.01, 2e3)])

    la = laplace_phase(wn*s/Q + wn**2, s**2 + wn*s/Q + wn**2, freqs, {fn:freq, d:damp})
    za = z_phase(*shock._absolute_acceleration_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)


@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_pseudovelocity_amp(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -ωₙ
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q
    With the amplitude response of:
                  |a₂(ωᵢ*j)|
    |G(ωᵢ*j)|  = ------------
                  |a₁(ωᵢ*j)|

    """
    dt = 1e-4
    omega = 2 * np.pi * freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_amplitude(-wn, s**2 + wn*s/Q + wn**2, freqs, {fn: freq, d: damp})
    za = z_amplitude(*shock._pseudo_velocity_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)

@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_pseudovelocity_phase(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -ωₙ
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q

    With the phase response of:
    ∠G(ωᵢ*j) = ∠a₂(ωᵢ*j) - ∠a₁(ωᵢ*j)
    """
    dt = 1e-4
    omega = 2*np.pi*freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_phase(-wn, s**2 + wn*s/Q + wn**2, freqs, {fn:freq, d:damp})
    za = z_phase(*shock._pseudo_velocity_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)


@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_eq_static_accel_amp(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -ωₙ²
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q
    With the amplitude response of:
                  |a₂(ωᵢ*j)|
    |G(ωᵢ*j)|  = ------------
                  |a₁(ωᵢ*j)|

    """
    dt = 1e-4
    omega = 2 * np.pi * freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_amplitude(-wn**2, s**2 + wn*s/Q + wn**2, freqs, {fn: freq, d: damp})
    za = z_amplitude(*shock._relative_displacement_static_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)

@hyp.given(
        freq=hyp_st.floats(12.5, 1200),
        damp=hyp_st.floats(*DAMP_RANGE, exclude_max=True),
        )
def test_eq_static_accel_phase(freq, damp):
    """
    Laplace domain transfer function:
           a₂(s)        -ωₙ²
    G(s) = ----- = ----------------
           a₁(s)         ωₙ*s
                   s² + ----- + ωₙ²
                          Q

    With the phase response of:
    ∠G(ωᵢ*j) = ∠a₂(ωᵢ*j) - ∠a₁(ωᵢ*j)
    """
    dt = 1e-4
    omega = 2*np.pi*freq

    freqs = np.geomspace(1e-1, 1000, 10000)

    la = laplace_phase(-wn**2, s**2 + wn*s/Q + wn**2, freqs, {fn:freq, d:damp})
    za = z_phase(*shock._relative_displacement_static_coefficients(omega, 1/(2*damp), dt), freqs, dt)

    npt.assert_allclose(za, la, rtol=.1, atol=1e-6)


@hyp.given(
    df_accel=hyp_np.arrays(
        dtype=np.float64,
        shape=(40, 2),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(
        lambda array: pd.DataFrame(
            np.concatenate([array, np.zeros_like(array)], axis=0),
            index=np.arange(2 * array.shape[0]) * 1e-4,
        )
    ),
    freq=hyp_st.floats(1, 20),
    damp=hyp_st.floats(1e-25, 1, exclude_max=True),
    mode=hyp_st.sampled_from(["srs", "pvss"]),
    aggregate_axes_two_sided=hyp_st.sampled_from(
        [(False, False), (False, True), (True, False)]
    ),
)
def test_pseudo_velocity_zero_padding(
    df_accel, freq, damp, mode, aggregate_axes_two_sided
):
    aggregate_axes, two_sided = aggregate_axes_two_sided

    # Check that the padding is all zeros
    assert np.all(df_accel.iloc[40:] == 0)

    # First, we calculate the PVSS of the data as-is
    calc_result = shock.shock_spectrum(
        df_accel.iloc[:40],
        [freq],
        damp=damp,
        mode=mode,
        aggregate_axes=aggregate_axes,
        two_sided=two_sided,
    )

    # Now we re-run the PVSS on the full, zero-padded data
    calc_result_padded = shock.shock_spectrum(
        df_accel,
        [freq],
        damp=damp,
        mode=mode,
        aggregate_axes=aggregate_axes,
        two_sided=two_sided,
    )

    # If the calculation is correct, there should be *no* amount of zero-padding
    # that changes the result
    if two_sided:
        for i in range(2):
            pd.testing.assert_frame_equal(calc_result[i], calc_result_padded[i])
    else:
        pd.testing.assert_frame_equal(calc_result, calc_result_padded)


@hyp.given(
    df_pvss=hyp_np.arrays(
        dtype=np.float64,
        shape=(40, 2),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(1, 41))),
    damp=hyp_st.floats(1e-25, 0.2),
)
def test_enveloping_half_sine(df_pvss, damp):
    env_half_sine = shock.enveloping_half_sine(df_pvss, damp=damp)
    hyp.note(f"pulse amplitude: {env_half_sine.amplitude}")
    hyp.note(f"pulse duration: {env_half_sine.duration}")

    pulse = env_half_sine.to_time_series()
    pulse_pvss = shock.shock_spectrum(
        pulse, freqs=df_pvss.index, damp=damp, mode="pvss"
    )

    # This is an approximation -> give the result a fudge-factor for correctness
    assert (df_pvss / pulse_pvss).max().max() < 1.2


class TestHalfSineWavePulse:
    @pytest.mark.parametrize(
        "tstart, tstop, dt, tpulse, warning_type",
        [
            # dt warnings
            (None, None, 0.12, 0, None),  # dt < duration / 8 => OK
            (None, None, 0.13, 0, UserWarning),  # dt > duration / 8 => WARNING
            # trange warnings
            (None, 0.5, None, 0, UserWarning),  # trange[1] < t0 + duration => WARNING
            (0.5, None, None, 0, UserWarning),  # trange[0] > t0 => WARNING
            (0.5, None, None, 1, None),  # OK
        ],
    )
    def test_to_time_series_warnings(self, tstart, tstop, dt, tpulse, warning_type):
        env_half_sine = shock.HalfSineWavePulse(
            amplitude=pd.Series([1]),
            duration=pd.Series([1]),
        )

        if warning_type is None:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                env_half_sine.to_time_series(
                    tstart=tstart, tstop=tstop, dt=dt, tpulse=tpulse
                )
        else:
            with pytest.warns(warning_type):
                env_half_sine.to_time_series(
                    tstart=tstart, tstop=tstop, dt=dt, tpulse=tpulse
                )

    def test_tuple_like(self):
        env_half_sine = shock.HalfSineWavePulse(
            amplitude=mock.sentinel.amplitude,
            duration=mock.sentinel.duration,
        )

        ampl, T = env_half_sine
        assert ampl == mock.sentinel.amplitude
        assert T == mock.sentinel.duration

@pytest.fixture
def df_test():
    half = shock.HalfSineWavePulse(
        amplitude=pd.Series([100, 50]),
        duration=pd.Series([.1, .05])
    )

    df_accel = pd.concat([
        half.to_time_series(tstart=0, tstop=1),
        half.to_time_series(tstart=1, tstop=2),
    ])
    df_accel.columns = ['A', 'B']

    return df_accel


class TestRollingShockSpectrum:

    def test_even_slices(self, df_test):
        df_rolling_srs = shock.rolling_shock_spectrum(
            df_test,
            num_slices=2,
            add_resultant=False,
            init_freq=2
        )

        # Get the Individual SRS
        first_srs = shock.shock_spectrum(df_test[:1.0], init_freq=2)
        second_srs = shock.shock_spectrum(df_test[1.0:], init_freq=2)

        # Do Assertions
        npt.assert_almost_equal(
            df_rolling_srs[(df_rolling_srs.variable == 'A') & (df_rolling_srs.timestamp == 0.5)]['value'].to_numpy(),
            first_srs['A'].to_numpy())
        npt.assert_almost_equal(
            df_rolling_srs[(df_rolling_srs.variable == 'B') & (df_rolling_srs.timestamp == 0.5)]['value'].to_numpy(),
            first_srs['B'].to_numpy())
        npt.assert_almost_equal(
            df_rolling_srs[(df_rolling_srs.variable == 'A') & (df_rolling_srs.timestamp != 0.5)]['value'].to_numpy(),
            second_srs['A'].to_numpy())
        npt.assert_almost_equal(
            df_rolling_srs[(df_rolling_srs.variable == 'B') & (df_rolling_srs.timestamp != 0.5)]['value'].to_numpy(),
            second_srs['B'].to_numpy())

    def test_defined_slices(self, df_test):
        df_rolling_srs = shock.rolling_shock_spectrum(
            df_test,
            index_values=[1.0],
            bins_per_octave=6,
            slice_width=0.5,
            mode='pvss',
            init_freq=4
        )

        npt.assert_almost_equal(
            df_rolling_srs[df_rolling_srs.variable == 'Resultant']['value'].to_numpy(),
            shock.shock_spectrum(
                df_test[0.75:1.2499],
                bins_per_octave=6,
                mode='pvss',
                aggregate_axes=True,
                init_freq=4
            )['Resultant'].to_numpy()
        )

