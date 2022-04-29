from collections import namedtuple
import pathlib
import timeit
import textwrap

import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import numpy as np
import numpy.testing as npt
import pandas as pd

from endaq.calc import fft, stats, utils


class TestFFT:

    @pytest.mark.parametrize("normalization, output, n",
                             [
                                 (None,       None,        None),
                                 ("unit",     None,        None),
                                 ("forward",  None,        None),
                                 ("backward", None,        None),
                                 ("ortho",    None,        None),
                                 (None,       "magnitude", None),
                                 (None,       "angle",     None),
                                 (None,       "complex",   None),
                             ],
                             )
    def test_fft(self, normalization, output, n):

        data = np.array([0.0]*10 + [-0.5, -1.0, 1.0, 0.5] + [0.0]*10)

        fxx = np.fft.fftshift(np.fft.fft(data))

        n = len(data)

        scales = {"unit": 2./n, "forward": 1./n, "backward": 1., "ortho": np.sqrt(1./n)}

        scale = scales.get(normalization, 2./n)
        fxx *= scale

        if output == "angle":
            fxx = np.angle(fxx)
        elif output == "complex":
            pass
        elif output == "magnitude":
            fxx = np.abs(fxx)
        else:
            fxx = np.abs(fxx)

        df = pd.DataFrame(
            index=np.linspace(0, n - 1, n),
            data=data,
            columns=['A'],
        )

        npt.assert_almost_equal(fft.fft(df, norm=normalization, nfft=n, output=output)['A'], fxx)

    @pytest.mark.parametrize(
        "normalization, output, n, raises",
        [
            ("sideways", None,       None,   pytest.raises(ValueError)),
            (None,       "octonian", None,   pytest.raises(ValueError)),
            (None,       None,       -1,     pytest.raises(ValueError)),
            (None,       None,       0,      pytest.raises(ValueError)),
            (None,       None,       0.,     pytest.raises(TypeError)),
            (None,       None,       "five", pytest.raises(TypeError)),
        ],
    )
    def test_fft_exceptions(self, normalization, output, n, raises):

        df = pd.DataFrame(np.arange(10))

        with raises:

            fft.fft(df, norm=normalization, output=output, nfft=n)


class TestRFFT:

    @pytest.mark.parametrize("normalization, output, n",
                             [
                                 (None,       None,        None),
                                 ("unit",     None,        None),
                                 ("forward",  None,        None),
                                 ("backward", None,        None),
                                 ("ortho",    None,        None),
                                 (None,       "magnitude", None),
                                 (None,       "angle",     None),
                                 (None,       "complex",   None),
                             ],
                             )
    def test_rfft(self, normalization, output, n):

        data = np.array([0.0]*10 + [-0.5, -1.0, 1.0, 0.5] + [0.0]*10)

        fxx = np.fft.rfft(data)

        n = len(data)

        scales = {"unit": 2./n, "forward": 1./n, "backward": 1., "ortho": np.sqrt(1./n)}

        scale = scales.get(normalization, 2./n)
        fxx *= scale

        if output == "angle":
            fxx = np.angle(fxx)
        elif output == "complex":
            pass
        elif output == "magnitude":
            fxx = np.abs(fxx)
        else:
            fxx = np.abs(fxx)

        df = pd.DataFrame(
            index=np.linspace(0, n - 1, n),
            data=data,
            columns=['A'],
        )

        npt.assert_almost_equal(fft.rfft(df, norm=normalization, nfft=n, output=output)['A'], fxx)

    @pytest.mark.parametrize(
        "normalization, output, n, raises",
        [
            ("sideways", None,       None,   pytest.raises(ValueError)),
            (None,       "octonian", None,   pytest.raises(ValueError)),
            (None,       None,       -1,     pytest.raises(ValueError)),
            (None,       None,       0,      pytest.raises(ValueError)),
            (None,       None,       0.,     pytest.raises(TypeError)),
            (None,       None,       "five", pytest.raises(TypeError)),
        ],
    )
    def test_fft_exceptions(self, normalization, output, n, raises):

        df = pd.DataFrame(np.arange(10))

        with raises:
            fft.rfft(df, norm=normalization, output=output, nfft=n)


@pytest.fixture
def df_test():
    # Build Time Array
    time = np.arange(1000) / 1000

    # Build Dataframe with Noise
    df_accel = pd.DataFrame({
        'time': np.concatenate((time, time + 1.0)),
        'A': np.concatenate((
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 13 * time))
        )
    }).set_index('time')

    # Add Random
    df_accel.A = df_accel.A + np.random.random(2000) * 2 - 1

    return df_accel


class TestRollingFFT:

    def test_using_spectrogram(self, df_test):
        df_rolling_fft = fft.rolling_fft(
            df_test,
            num_slices=2,
            add_resultant=False
        )
        times = df_rolling_fft.timestamp.unique()

        npt.assert_almost_equal(
            df_rolling_fft[df_rolling_fft.timestamp == times[0]].value.to_numpy(),
            fft.aggregate_fft(df_test[:1.0])['A'].to_numpy())
        npt.assert_almost_equal(
            df_rolling_fft[df_rolling_fft.timestamp == times[1]].value.to_numpy(),
            fft.aggregate_fft(df_test[1.0:])['A'].to_numpy())

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_defined_slices(self, df_test):
        df_rolling_fft = fft.rolling_fft(
            df_test,
            index_values=[1.0, 1.5],
            slice_width=0.5,
            add_resultant=False,
        )

        # Do Assertions
        npt.assert_almost_equal(
            df_rolling_fft[df_rolling_fft.timestamp == 1.0].value.to_numpy(),
            fft.aggregate_fft(df_test.iloc[750:1250])['A'].to_numpy())
        npt.assert_almost_equal(
            df_rolling_fft[df_rolling_fft.timestamp == 1.5].value.to_numpy(),
            fft.aggregate_fft(df_test.iloc[1250:1750])['A'].to_numpy())
