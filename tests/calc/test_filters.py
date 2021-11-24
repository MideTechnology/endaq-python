import numpy as np
import pandas as pd

from endaq.calc import filters


def test_highpass():
    # Generate input data
    x = np.zeros(1000)
    x[300] = 1
    fs = 100
    fs_cutoff = 5

    # Validate input data
    f = np.fft.rfftfreq(len(x), d=1 / fs)
    Fx = np.fft.rfft(x)
    assert np.allclose(np.abs(Fx), 1)

    # Generate output data
    x_filt = (
        filters.butterworth(
            pd.DataFrame(
                x, index=(np.arange(len(x)) / fs) * np.timedelta64(10 ** 9, "ns")
            ),
            low_cutoff=fs_cutoff,
            high_cutoff=None,
        )
        .to_numpy()
        .flatten()
    )
    assert x_filt.shape == x.shape

    # Validate output data
    Fx_filt = np.fft.rfft(x_filt)
    assert (
        np.abs(Fx_filt[f == 0]) < 1e-9
    )  # diverges from other trends at f=0 -> test explicitly here

    f_cut_lower = fs_cutoff / 2
    f_cut_upper = fs_cutoff * 2
    i_f_cut_lower = np.flatnonzero(np.diff(f > f_cut_lower, prepend=0))[0]
    i_f_cut_upper = np.flatnonzero(np.diff(f > f_cut_upper, prepend=0))[0]

    Fx_filt_dB = 10 * np.log10(np.abs(Fx_filt))
    assert np.allclose(Fx_filt_dB[f >= f_cut_upper], 0, atol=0.1)

    f_decade = np.log10(f)
    # diverges from trend at f=0
    Fx_filt_dB_diff = np.diff(Fx_filt_dB[f != 0]) / np.diff(f_decade[f != 0])
    # pre-cutoff ramps up at constant rate
    assert np.allclose(Fx_filt_dB_diff[: i_f_cut_lower - 1], 10 * 6, atol=1)
    # post-cutoff constant at 0dB
    assert np.allclose(Fx_filt_dB_diff[i_f_cut_upper:], 0, atol=1)
    # in-between slows down from ramp-up to constant
    assert np.all(np.diff(Fx_filt_dB_diff) < 1e-2)

    mask_thresh = np.abs(Fx_filt) > 1e-8
    angle_change = np.angle(Fx_filt[mask_thresh]) - np.angle(Fx[mask_thresh])
    angle_change_0centered = (angle_change + np.pi) % (2 * np.pi) - np.pi
    # 0-phase offset with bidirectional filter
    assert np.allclose(angle_change_0centered, 0)
