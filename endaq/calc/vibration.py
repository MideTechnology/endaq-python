from typing import Optional

import pandas as pd

from .psd import welch


def _vrs(pxx, delta_f, freqs):
    pass


def vibration_spectrum(
        accel: pd.DataFrame,
        bin_width: float = 1.0,
        damp: float = 0.05,
        ) -> pd.DataFrame:
    """

    :param accel:
    :param nbins:
    :param damp:
    :return:
    """

    if bin_width <= 0:
        raise ValueError(f'bin_width must be greater than 0, got {bin_width}')

    pxx = welch(accel, bin_width=bin_width)

    pass
