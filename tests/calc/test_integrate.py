import pytest
import numpy as np
import pandas as pd
import sympy as sp

from endaq.calc import integrate


def test_integrate():
    """Test `_integrate` via differentiation."""
    n = 20
    array = np.array([sp.symbols(f"x:{n}"), sp.symbols(f"y:{n}")]).T
    dt = sp.symbols("dt")

    # Ensure derivative looks correct
    calc_result = integrate._integrate(
        pd.DataFrame(array, index=dt * np.arange(len(array)))
    )
    expt_result_diff = 0.5 * dt * (array[:-1] + array[1:])
    assert np.all(np.diff(calc_result, axis=0) - expt_result_diff == 0)

    # Ensure offset results in zero-mean data
    # Note: symbols cannot be directly tested, since scalar factors are floats
    assert calc_result.to_numpy().mean().subs(
        [(dt, 1)] + list(zip(array.flatten(), np.ones(array.size)))
    ) == pytest.approx(0)


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
