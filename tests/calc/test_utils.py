import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(__file__, '..', '..', '..')))

from endaq.calc import utils


@hyp_st.composite
def logfreq_input(draw):
    dt = draw(hyp_st.floats(1e-7, 1e7))
    n = 40
    fs = 1 / dt
    df = fs / n

    dframe = pd.DataFrame(np.zeros((n, 2)), index=dt * np.arange(n))
    init_freq = draw(hyp_st.floats(df, fs / 2, exclude_max=True))
    bins_per_octave = draw(hyp_st.floats(0.1, 10))

    return dframe, init_freq, bins_per_octave


@hyp.given(input_vars=logfreq_input())
@pytest.mark.filterwarnings("ignore:the data's duration is too short:RuntimeWarning")
def test_logfreqs(input_vars):
    (dframe, init_freq, bins_per_octave) = input_vars

    n = len(dframe.index)
    dt = (dframe.index[-1] - dframe.index[0]) / (n - 1)
    fs = 1 / dt
    df = fs / n

    calc_result = utils.logfreqs(dframe, init_freq, bins_per_octave)

    assert calc_result.ndim == 1
    if init_freq > fs / 2:
        assert len(calc_result) == 0
        return
    assert calc_result[0] == pytest.approx(init_freq)
    assert fs / 2 ** (1 + 1 / bins_per_octave) < calc_result[-1] < fs / 2
    np.testing.assert_allclose(np.diff(np.log2(calc_result)), 1 / bins_per_octave)


@hyp.given(value=hyp_st.floats(1e-7, 1e7))
def test_to_dB_ref(value):
    assert utils.to_dB(value, value) == 0


@hyp.given(
    value=hyp_st.floats(1e-7, 1e7),
    reference=hyp_st.floats(1e-7, 1e7),
    squared=hyp_st.booleans(),
)
def test_to_dB_scale(value, reference, squared):
    scale = 10 if squared else 20
    assert utils.to_dB(10 * value, reference, squared) == pytest.approx(
        scale + utils.to_dB(value, reference, squared)
    )


def test_uniform_resample_is_uniform_for_datetime64ns():
    """
    Tests if the resample function for np.datetime64[ns] dtype arrays will produce non-uniform timestamps
    """
    times = np.arange('2021-02', '2021-03', np.timedelta64(1, 's'), dtype='datetime64[ns]')
    df = pd.DataFrame(np.arange(len(times)), index=times)
    assert len(np.unique(np.diff(utils.resample(df).index))) == 1


def test_rolling_slice_definitions():
    # Build dataframe with 1 second of data
    df = pd.DataFrame({
        'time': np.arange(1000) / 1000,
        'A': np.ones(1000)
    }).set_index('time')

    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df=df,
        num_slices=2
    )
    assert slice_width == 0.5

    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df=df,
        index_values=[0.1, 0.9]
    )
    assert indexes[1] == 900

    df.index = pd.to_datetime(df.index, unit='s')
    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df=df,
    )
    assert num == 5

    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df=df,
        index_values=pd.DatetimeIndex(['1970-01-01 00:00:00.9951'])
    )
    assert indexes[0] == 995


def test_convert_units():
    assert utils.convert_units('in', 'mm') == 25.4

    df = pd.DataFrame({'Val': [-40, 0, 10]})
    np.testing.assert_allclose(utils.convert_units('degC', 'degF', df).Val[0], -40)
    np.testing.assert_allclose(utils.convert_units('degC', 'degF', df).Val[1], 32)


def test_to_altitude():
    """
    Tests the accuracy of the to_altitude function, which converts air pressure
    to altitude. These tests only cover measurements BELOW the stratosphere.
    """
    # Pressure Data CSV File --> DataFrame
    df = pd.read_csv("tests/calc/csv_to_df/default_sea_lvl.csv")

    # Meters
    # DataFrame 1; Default settings:
    def_key = [0.00, 110.88, 540.34, 988.50, 1457.30, 1948.99, 2466.23,
                3012.18, 3590.69, 4206.43, 4865.22, 5574.44, 6343.62, 7185.44,
                8117.27, 9163.96, 10362.95]
    default_df = utils.to_altitude(df=df)
    altitude_list_default = default_df['Altitude (m)'].tolist()
    altitude_list_default = [round(num, 2) for num in altitude_list_default]
    assert (altitude_list_default == def_key
            ), "Equation is not accurate with default settings."


    # DataFrame 2; Different base temperature:
    temp_key = [0.00, 116.66, 568.47, 1039.96, 1533.16, 2050.45, 2594.61,
                3168.99, 3777.61, 4425.40, 5118.48, 5864.62, 6673.85, 7559.48,
                8539.82, 9641.00, 10902.40]
    diff_temp_df = utils.to_altitude(df=df, base_temp=30)
    altitude_list_diff_temp = diff_temp_df['Altitude (m)'].tolist()
    altitude_list_diff_temp = [round(num, 2) for num in altitude_list_diff_temp]
    assert (altitude_list_diff_temp == temp_key
            ), "Equation is not accurate with non-default base temperature."

    # DataFrame 3: Different base pressure:
    press_key = [-111.16, 0, 430.53, 879.82, 1349.79, 1842.71, 2361.25, 2908.57,
                 3488.53, 4105.81, 4766.26, 5477.25, 6248.37, 7092.29, 8026.46,
                 9075.77, 10277.77]
    diff_press_df = utils.to_altitude(df=df, base_press=100000)
    altitude_list_diff_press = diff_press_df['Altitude (m)'].tolist()
    altitude_list_diff_press = [round(num, 2) for num in altitude_list_diff_press]
    assert (altitude_list_diff_press == press_key
            ), "Equation is not accurate with non-default base pressure."

    # DataFrame 4: Different base temperature and pressure:
    temp_press_key = [-116.95, 0.00, 452.94, 925.62, 1420.06, 1938.64, 2484.17,
                      3059.98, 3670.13, 4319.54, 5014.37, 5762.38, 6573.63,
                      7461.49, 8444.29, 9548.22, 10812.79]
    diff_temp_and_press_df = utils.to_altitude(df=df, base_temp=30,
                                               base_press=100000)
    altitude_list_diff_temp_press = diff_temp_and_press_df['Altitude (m)'].tolist()
    altitude_list_diff_temp_press = [round(num, 2) for num in 
                                     altitude_list_diff_temp_press]
    assert (altitude_list_diff_temp_press == temp_press_key
            ), "Equation is not accurate with non-default base temperature and pressure."

    # Feet --> Meters --> Feet
    # DataFrame 1; Default settings; Units = Feet:
    def_key = [0.00, 363.79, 1772.76, 3243.11, 4781.17, 6394.32, 8091.29,
               9882.49, 11780.47, 13800.61, 15962.0, 18288.84, 20812.4,
               23574.27, 26631.46, 30065.48, 33999.16]
    default_df = utils.to_altitude(df=df, units='ft')
    altitude_list_default = default_df['Altitude (ft)'].tolist()
    altitude_list_default = [round(num, 2) for num in altitude_list_default]
    assert (altitude_list_default == def_key
            ), "Equation is not accurate for units='ft'."
