import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np
import pytest

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy import signal

from endaq.calc import stats, utils, filters, integrate, psd, shock

# If `True`, tests will display plots in the browser. Causes issues
# with Windows tests in GitHub Actions (possibly temporary).
# TODO: Make this based on whether test is running locally or in a GHA.
DISPLAY_PLOTS = False

@hyp.given(
    df=hyp_np.arrays(
        dtype=np.float64,
        shape=(40, 2),
        elements=hyp_st.floats(-1e20, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(40) * 1e-4)),
    window_len=hyp_st.integers(1, 40),
)
def test_rolling_rms(df, window_len):
    calc_result = stats.rolling_rms(df, window_len)
    expt_result = df.rolling(window_len).apply(stats.rms)
    pd.testing.assert_frame_equal(calc_result, expt_result)

@pytest.fixture
def generate_time_dataframe():
    time = np.arange(1000) / 1000
    df_accel = pd.DataFrame({
        'time': np.concatenate((time, time + 1.0, time + 2.0)),
        'A': np.random.laplace(0, 1, 3000),
        'B': np.random.laplace(0, 1, 3000)
    }).set_index('time')

    return df_accel


@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
def test_shock_vibe_metrics(generate_time_dataframe):
    tukey_percent = 0.1
    highpass_cutoff = 5.0
    zero = 'start'
    bins_per_octave = 12
    init_freq = 2
    damp = 0.05
    accel_units = 'gravity'
    disp_units = 'feet'
    accel_2_disp = utils.convert_units(units_in=accel_units, units_out=disp_units + '/s^2')

    # Calculate Metrics Using Function
    metrics = stats.shock_vibe_metrics(
        generate_time_dataframe, include_resultant=True, include_pseudo_velocity=True,
        tukey_percent=tukey_percent, highpass_cutoff=highpass_cutoff,
        display_plots=DISPLAY_PLOTS, damp=damp, init_freq=init_freq,
        bins_per_octave=bins_per_octave, accel_units=accel_units, disp_units=disp_units)

    # Do Integration Related Calculations
    df = filters.butterworth(
        generate_time_dataframe - generate_time_dataframe.median(),
        low_cutoff=highpass_cutoff, tukey_percent=tukey_percent)

    [accel, vel, disp] = integrate.integrals(
        df * accel_2_disp, n=2, tukey_percent=tukey_percent, highpass_cutoff=highpass_cutoff, zero=zero)

    rms_start = int(tukey_percent / 2 * df.shape[0])
    rms_end = df.shape[0] - rms_start
    disp_rms = disp.iloc[rms_start:rms_end].pow(2).mean() ** 0.5

    np.testing.assert_almost_equal(
        disp_rms['A'],
        metrics[(metrics.variable == 'A') & (metrics.calculation == 'RMS Displacement')].value[0]
    )

    np.testing.assert_almost_equal(
        disp.pow(2).sum(axis=1).pow(0.5).abs().max(),
        metrics[(metrics.variable == 'Resultant') & (metrics.calculation == 'Peak Absolute Displacement')
                ].value[0]
    )

    # Do Frequency Related Calculations
    fs = 1 / utils.sample_spacing(df)
    parseval = psd.welch(df, scaling='parseval', bin_width=fs / len(df))
    freq_splits = [0, 65, 300]
    rms_psd_breaks = psd.to_jagged(parseval, freq_splits, agg='sum') ** 0.5

    np.testing.assert_almost_equal(
        rms_psd_breaks['B'].to_numpy()[0],
        metrics[(metrics.variable == 'B') & (metrics.calculation == 'RMS from 0 to 65')
                ].value.to_numpy()[0]
    )

    # Do PVSS
    pvss = shock.shock_spectrum(df * accel_2_disp, init_freq=init_freq,
                                bins_per_octave=bins_per_octave, damp=damp,
                                mode='pvss', aggregate_axes=True)['Resultant']

    np.testing.assert_almost_equal(
        pvss.max(),
        metrics[(metrics.variable == 'Resultant') & (metrics.calculation == 'Peak Pseudo Velocity')
                ].value.to_numpy()[0]
    )


def test_find_peak(generate_time_dataframe):
    df = generate_time_dataframe
    time_distance = 0.5
    distance = int(time_distance / utils.sample_spacing(df))
    resultant = df.pow(2).sum(axis=1).pow(0.5).to_numpy()

    np.testing.assert_almost_equal(
        signal.find_peaks(
            resultant,
            distance=distance,
            height=resultant*0.1,
        )[0],
        stats.find_peaks(
            df,
            display_plots=DISPLAY_PLOTS,  # Was True, but hangs Windows GHA tests
            time_distance=time_distance,
            threshold_multiplier=0.1,
            threshold_reference="peak",
            add_resultant=True,
            use_abs=False
        )
    )

    np.testing.assert_almost_equal(
        signal.find_peaks(
            df.abs().max(axis=1).to_numpy(),
            distance=distance,
            height=(df.pow(2).mean() ** 0.5).max() * 4.0,
        )[0],
        stats.find_peaks(
            df,
            display_plots=DISPLAY_PLOTS,  # Was True, but hangs Windows GHA tests
            time_distance=time_distance,
            threshold_multiplier=4.0,
            threshold_reference="rms",
            add_resultant=False,
            use_abs=True
        )
    )


class TestRollingMetrics:

    def test_num_slices(self, generate_time_dataframe):
        df = generate_time_dataframe
        df_rolling_metrics = stats.rolling_metrics(
            df,
            num_slices=2
        )
        times = df_rolling_metrics.timestamp.unique()

        pd.testing.assert_frame_equal(
            df_rolling_metrics[df_rolling_metrics.timestamp == times[0]][['variable', 'value', 'calculation', 'units']],
            stats.shock_vibe_metrics(df[:1.4999])
        )
        pd.testing.assert_frame_equal(
            df_rolling_metrics[df_rolling_metrics.timestamp == times[1]][['variable', 'value', 'calculation', 'units']],
            stats.shock_vibe_metrics(df[1.5:])
        )

    def test_defined_slices(self, generate_time_dataframe):
        df = generate_time_dataframe
        df_rolling_metrics = stats.rolling_metrics(
            df,
            num_slices=2,
            index_values=[1.0, 1.5],
            slice_width=0.5,
        )

        pd.testing.assert_frame_equal(
            df_rolling_metrics[df_rolling_metrics.timestamp == 1.0][['variable', 'value', 'calculation', 'units']],
            stats.shock_vibe_metrics(df.iloc[750:1250])
        )
        pd.testing.assert_frame_equal(
            df_rolling_metrics[df_rolling_metrics.timestamp == 1.5][['variable', 'value', 'calculation', 'units']],
            stats.shock_vibe_metrics(df.iloc[1250:1750])
        )
