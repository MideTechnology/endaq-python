import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd
import sympy as sp

import endaq
from endaq.calc import integrate


@hyp.given(
    df=hyp_np.arrays(
        elements=hyp_st.floats(-1e7, 1e7),
        shape=(20, 2),
        dtype=np.float64,
    ).map(lambda array: pd.DataFrame(array, index=0.2 * np.arange(20))),
)
def test_integrate(df):
    """Test `_integrate` via differentiation."""
    dt = endaq.calc.sample_spacing(df)

    # Ensure derivative looks correct
    calc_result = integrate._integrate(df)
    expt_result_diff = 0.5 * dt * (df.to_numpy()[:-1] + df.to_numpy()[1:])
    assert np.diff(calc_result.to_numpy(), axis=0) == pytest.approx(expt_result_diff)

    # Ensure offset results in zero-mean data
    # Note: symbols cannot be directly tested, since scalar factors are floats
    np.testing.assert_allclose(
        calc_result.mean(), 0, atol=1e-7 * df.abs().mean().mean()
    )


def test_integrals():
    n = 20
    array = np.array([sp.symbols(f"x:{n}"), sp.symbols(f"y:{n}")]).T
    dt = sp.symbols("dt")
    df = pd.DataFrame(array, index=dt * np.arange(len(array)))

    calc_result = integrate.integrals(df, n=2)

    assert len(calc_result) == 3
    assert np.all(calc_result[0] == df)
    for dx_dt, x in zip(calc_result[:-1], calc_result[1:]):
        assert np.all(x == integrate._integrate(dx_dt))
