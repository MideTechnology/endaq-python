# -*- coding: utf-8 -*-

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd

from endaq.calc import shock


@hyp.given(
    freq=hyp_st.floats(12.5, 1000),
    damp=hyp_st.floats(0, 1, exclude_max=True),
)
def test_rel_displ(freq, damp):
    """
    This test uses a step-function input acceleration. In a SDOF spring system,
    the spring should be relaxed in the first portion where `a(t < t0) = 0`.
    Once the acceleration flips on (`a(t > t0) = 1`), the mass should begin to
    oscillate.

    (This scenario is mathematically identical to having the mass pulled out
    some distance and held steady with a constant force at `t=0`, then
    releasing the mass at `t > t0` and letting it oscillate freely.)

    This system is tested over a handful of different oscillation parameter
    (i.e., frequency & damping rate) configurations.
    """
    # Data parameters
    signal = np.zeros(1000, dtype=float)
    signal[200:] = 1
    fs = 10 ** 4  # Hz
    # Other parameters
    omega = 2 * np.pi * freq

    # Calculate result
    calc_result = (
        shock.rel_displ(
            pd.DataFrame(signal, index=np.arange(len(signal)) / fs),
            omega=omega,
            damp=damp,
        )
        .to_numpy()
        .flatten()
    )

    # Calculate expected result
    t = np.arange(1, 801) / fs
    atten = omega * (-damp + 1j * np.sqrt(1 - damp ** 2))
    assert np.angle(atten) == pytest.approx(np.arccos(-damp))
    assert np.abs(atten) == pytest.approx(omega)

    # γ = -ζ + i√(1 - ζ²)
    # h(t) = (1/Im{γ}) Im{exp(γωt)}
    # u(t) := Heaviside Step Function
    # -> result = {h * u}(t) = ∫h(t) dt
    #     = C + (1 / Im{γ}) Im{1/γω exp(γωt)}
    expt_result = np.zeros_like(signal)
    expt_result[200:] = (-1 / np.imag(atten)) * np.imag(
        np.exp(t * atten) / atten
    ) - 1 / omega ** 2

    # Test results
    assert np.allclose(calc_result, expt_result)


@hyp.given(
    freq=hyp_st.floats(12.5, 1000),
    damp=hyp_st.floats(0, 1, exclude_max=True),
)
def test_abs_accel(freq, damp):
    """
    This test uses a step-function input acceleration. In a SDOF spring system,
    the spring should be relaxed in the first portion where `a(t < t0) = 0`.
    Once the acceleration flips on (`a(t > t0) = 1`), the mass should begin to
    oscillate.

    (This scenario is mathematically identical to having the mass pulled out
    some distance and held steady with a constant force at `t=0`, then
    releasing the mass at `t > t0` and letting it oscillate freely.)

    This system is tested over a handful of different oscillation parameter
    (i.e., frequency & damping rate) configurations.
    """
    # Data parameters
    signal = np.zeros(1000, dtype=float)
    signal[200:] = 1
    fs = 10 ** 4  # Hz
    # Other parameters
    omega = 2 * np.pi * freq

    # Calculate result
    calc_result = (
        shock.abs_accel(
            pd.DataFrame(signal, index=np.arange(len(signal)) / fs),
            omega=omega,
            damp=damp,
        )
        .to_numpy()
        .flatten()
    )

    # Calculate expected result
    t = np.arange(1, 801) / fs
    atten = omega * (-damp + 1j * np.sqrt(1 - damp ** 2))
    assert np.angle(atten) == pytest.approx(np.arccos(-damp))
    assert np.abs(atten) == pytest.approx(omega)

    # γ = -ζ + i√(1 - ζ²)
    # h(t) = (2ζω) Re{exp(γωt)} + ((1 - 2ζ)/Im{γ}) Im{exp(γωt)}
    # u(t) := Heaviside Step Function
    # -> result = {h * u}(t) = ∫h(t) dt
    #     = C + (2ζω) Re{1/γω exp(γωt)} + ((1 - 2ζ)/Im{γ}) Im{1/γω exp(γωt)}
    expt_result = np.zeros_like(signal)
    expt_result[200:] = (
        omega
        * (
            np.real(np.exp(t * atten) / atten) * 2 * damp
            + np.imag(np.exp(t * atten) / atten)
            * (1 - 2 * damp ** 2)
            / np.sqrt(1 - damp ** 2)
        )
        + 1
    )

    # Test results
    assert np.allclose(calc_result, expt_result)


@hyp.given(
    df_accel=hyp_np.arrays(
        dtype=np.float64,
        shape=(40, 2),
        elements=hyp_st.floats(1e-20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(40) * 1e-4)),
    freq=hyp_st.floats(1, 20),
    damp=hyp_st.floats(0, 1, exclude_max=True),
    mode=hyp_st.sampled_from(["srs", "pvss"]),
    aggregate_axes_two_sided=hyp_st.sampled_from(
        [(False, False), (False, True), (True, False)]
    ),
    factor=hyp_st.floats(-1e2, 1e2),
)
def test_shock_spectrum_linearity(
    df_accel,
    freq,
    damp,
    mode,
    aggregate_axes_two_sided,
    factor,
):
    aggregate_axes, two_sided = aggregate_axes_two_sided

    calc_result = shock.shock_spectrum(
        df_accel,
        [freq],
        damp=damp,
        mode=mode,
        aggregate_axes=aggregate_axes,
        two_sided=two_sided,
    )
    calc_result_prescale = shock.shock_spectrum(
        factor * df_accel,
        [freq],
        damp=damp,
        mode=mode,
        aggregate_axes=aggregate_axes,
        two_sided=two_sided,
    )
    if two_sided:
        calc_result_postscale = tuple(
            df_calc * np.abs(factor) for df_calc in calc_result
        )[slice(None) if factor >= 0 else slice(None, None, -1)]
        for prescale, postscale in zip(calc_result_prescale, calc_result_postscale):
            pd.testing.assert_frame_equal(prescale, postscale)
    else:
        pd.testing.assert_frame_equal(
            calc_result_prescale,
            calc_result.mul(factor).abs(),
        )


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
    damp=hyp_st.floats(0, 1, exclude_max=True),
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
    damp=hyp_st.floats(0, 0.2),
)
@hyp.settings(deadline=None)  # this test tends to timeout
def test_enveloping_half_sine(df_pvss, damp):
    ampl, T = shock.enveloping_half_sine(df_pvss, damp=damp)
    hyp.note(f"pulse amplitude: {ampl}")
    hyp.note(f"pulse duration: {T}")

    ampl = ampl[0]
    T = T[0]

    dt = min(
        1 / (2 * df_pvss.index[-1]), T / 20
    )  # guarantee sufficient # of samples to represent pulse
    fs = 1 / dt
    times = np.arange(int(fs * (T + 1 / df_pvss.index[0]))) / fs
    pulse = np.zeros_like(times)
    pulse[: int(T * fs)] = ampl * np.sin((np.pi / T) * times[: int(T * fs)])
    pulse_pvss = shock.shock_spectrum(
        pd.DataFrame(pulse, index=times), freqs=df_pvss.index, damp=damp, mode="pvss"
    )

    # This is an approximation -> give the result a fudge-factor for correctness
    assert (df_pvss / pulse_pvss).max().max() < 1.2
