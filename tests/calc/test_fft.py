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
