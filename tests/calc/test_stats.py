import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd

from endaq.calc import stats


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
