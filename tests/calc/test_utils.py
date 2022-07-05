import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st

import numpy as np
import pandas as pd

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
