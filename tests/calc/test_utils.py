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
