from collections import namedtuple
import pathlib
import timeit
import textwrap

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import pandas as pd

from endaq.calc import psd, stats, utils


@hyp.given(
    df=hyp_np.arrays(
        dtype=np.float64,
        shape=(200,),
        elements=hyp_st.floats(
            # leave at least half the bits of precision (52 / 2) in the
            # mean-subtracted result
            1,
            1e26,
        ),
    )
    .map(
        lambda array: (array - array.mean(keepdims=True))
        # V this pushes the zero-mean'd values away from zero
        * 2 ** (np.finfo(np.float64).minexp // 2)
    )
    .map(lambda array: pd.DataFrame(array, index=np.arange(len(array)) * 1e-1)),
)
def test_welch_parseval(df):
    """
    Test to confirm that `scaling="parseval"` maintains consistency with the
    time-domain RMS.
    """
    df_psd = psd.welch(df, bin_width=1, scaling="parseval")
    assert df_psd.to_numpy().sum() == pytest.approx(stats.rms(df.to_numpy()) ** 2)


@hyp.given(
    psd_df=hyp_np.arrays(
        dtype=np.float64,
        shape=(20, 3),
        elements=hyp_st.floats(0, 1e20),
    ).map(lambda array: pd.DataFrame(array, index=np.arange(len(array)) * 10)),
    freq_splits=hyp_np.arrays(
        dtype=np.float64,
        shape=(8,),
        elements=hyp_st.floats(0, 200, exclude_min=True),
        unique=True,
    ).map(lambda array: np.sort(array)),
)
@pytest.mark.parametrize(
    "agg1, agg2",
    [
        ("mean", lambda x, axis=-1: np.nan_to_num(np.mean(x, axis=axis))),
        ("sum", np.sum),
    ],
)
@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
def test_to_jagged_modes(psd_df, freq_splits, agg1, agg2):
    """ Test `to_jagged(..., mode='mean')` against the equivalent `mode=np.mean`. """
    result1 = psd.to_jagged(psd_df, freq_splits, agg=agg1)
    result2 = psd.to_jagged(psd_df, freq_splits, agg=agg2)

    assert np.all(result1.index == result2.index)
    np.testing.assert_allclose(
        result1.to_numpy(),
        result2.to_numpy(),
        atol=psd_df.min().min() * 1e-7,
    )


@pytest.mark.skip(
    reason="timing test -> does not enforce functionality; takes too long"
)
def test_to_jagged_mode_times():
    """
    Check that a situation exists where the histogram method is more
    performant.
    """
    setup = textwrap.dedent(
        """
        from endaq.calc import psd
        import numpy as np
        import pandas as pd

        n = 10 ** 4

        axis = -1
        psd_array = np.random.random((3, n))
        f = np.arange(n) / 3
        psd_df = pd.DataFrame(psd_array.T, index=f)
        #freq_splits = np.logspace(0, np.log2(n), num=100, base=2)
        freq_splits = f[1:-1]
        """
    )

    t_direct = timeit.timeit(
        "psd.to_jagged(psd_df, freq_splits, agg=np.sum)",
        setup=setup,
        number=3,
    )
    t_hist = timeit.timeit(
        "psd.to_jagged(psd_df, freq_splits, agg='sum')",
        setup=setup,
        number=3,
    )

    print(f"direct form time: {t_direct}")
    print(f"histogram time: {t_hist}")
    assert t_hist < t_direct


_TestStruct = namedtuple("_TestStruct", "psd_df, agg, expt_f, expt_array")


@pytest.mark.parametrize(
    ", ".join(_TestStruct._fields),
    [
        _TestStruct(
            psd_df=pd.DataFrame([1, 0, 0, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 1, 0, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[1, 0, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 1, 0, 0, 0, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 1, 0, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 0, 1, 1, 1, 0, 0]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 3, 0],
        ),
        _TestStruct(
            psd_df=pd.DataFrame([0, 0, 0, 0, 0, 0, 1, 1]),
            agg="sum",
            expt_f=[1, 2, 4, 8],
            expt_array=[0, 0, 0, 2],
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
def test_to_octave(psd_df, agg, expt_f, expt_array):
    calc_df = psd.to_octave(psd_df, fstart=1, octave_bins=1, agg=agg)
    assert calc_df.index.to_numpy().tolist() == expt_f
    assert calc_df.to_numpy().flatten().tolist() == expt_array


@pytest.fixture
def df_test():
    # Build Time Array
    time = np.arange(1000) / 1000

    # Build Dataframe with Noise
    df_accel = pd.DataFrame({
        'time': np.concatenate((time, time + 1.0)),
        'A': np.random.random(2000)
    }).set_index('time')

    return df_accel


@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
class TestRollingPSD:

    def test_using_spectrogram(self, df_test):
        df_rolling_psd = psd.rolling_psd(
            df_test,
            num_slices=2,
            add_resultant=False
        )
        times = df_rolling_psd.timestamp.unique()

        np.testing.assert_almost_equal(
            df_rolling_psd[df_rolling_psd.timestamp == times[0]].value.to_numpy(),
            psd.welch(df_test[:1.0])['A'].to_numpy())
        np.testing.assert_almost_equal(
            df_rolling_psd[df_rolling_psd.timestamp == times[1]].value.to_numpy(),
            psd.welch(df_test[1.0:])['A'].to_numpy())

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_defined_slices(self, df_test):
        df_rolling_psd = psd.rolling_psd(
            df_test,
            index_values=[1.0, 1.5],
            slice_width=0.5,
            add_resultant=False,
        )

        # Do Assertions
        np.testing.assert_almost_equal(
            df_rolling_psd[df_rolling_psd.timestamp == 1.0].value.to_numpy(),
            psd.welch(df_test.iloc[750:1250])['A'].to_numpy())
        np.testing.assert_almost_equal(
            df_rolling_psd[df_rolling_psd.timestamp == 1.5].value.to_numpy(),
            psd.welch(df_test.iloc[1250:1750])['A'].to_numpy())