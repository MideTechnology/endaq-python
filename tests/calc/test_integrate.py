import itertools

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
    zero=hyp_st.sampled_from(["start", "mean", "median"]),
)
def test_integrate(df, zero):
    """Test `_integrate` via differentiation."""
    dt = endaq.calc.sample_spacing(df)

    # Ensure derivative looks correct
    calc_result = integrate._integrate(df, zero)
    expt_result_diff = 0.5 * dt * (df.to_numpy()[:-1] + df.to_numpy()[1:])
    assert np.diff(calc_result.to_numpy(), axis=0) == pytest.approx(expt_result_diff)

    # Ensure offset results in zero-mean data
    # Note: symbols cannot be directly tested, since scalar factors are floats
    zero_quantity = {
        "start": calc_result.iloc[0],
        "mean": calc_result.mean(),
        "median": calc_result.median(),
    }[zero]
    np.testing.assert_allclose(zero_quantity, 0, atol=1e-7 * df.abs().mean().mean())


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


@hyp.given(
    df=hyp_np.arrays(
        elements=hyp_st.floats(-1e7, 1e7),
        shape=(20, 2),
        dtype=np.float64,
    ).map(lambda array: pd.DataFrame(array, index=0.2 * np.arange(20))),
    zero=hyp_st.sampled_from(["start", "mean", "median"]),
    highpass_cutoff=hyp_st.one_of(hyp_st.floats(0.25, 1), hyp_st.just(None)),
    tukey_percent=hyp_st.floats(0, 0.2),
)
def test_integrals_iter_vs_list(df, **kwargs):
    n = 3
    result1 = list(itertools.islice(integrate.iter_integrals(df, **kwargs), 2))
    result2 = integrate.integrals(df, n, **kwargs)

    for r1, r2 in zip(result1, result2):
        pd.testing.assert_frame_equal(r1, r2)
