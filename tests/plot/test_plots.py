import pytest

import numpy as np
import pandas as pd

from endaq import plot


def test_spectrums_over_time():
    freqs = np.arange(10)
    times = np.zeros_like(freqs)

    df = pd.DataFrame({
        'frequency (Hz)': np.concatenate((freqs, freqs, freqs)),
        'timestamp': np.concatenate((times, times + 30, times + 60)),
        'value': np.random.random(30)
    })
    df['variable'] = 'X'

    fig = plot.spectrum_over_time(df, 'Heatmap')
    assert fig['data'][0]['type'] == 'heatmap'

    fig = plot.spectrum_over_time(df, 'Surface')
    assert fig['data'][0]['type'] == 'surface'

    fig = plot.spectrum_over_time(df, 'Waterfall')
    assert fig['data'][0]['type'] == 'scatter3d'

    fig = plot.spectrum_over_time(df, 'Animation')
    assert len(fig['data']) == 4

    fig = plot.spectrum_over_time(df, 'Lines')
    assert len(fig['data']) == 9

    fig = plot.spectrum_over_time(df, 'Peak')
    assert fig['data'][0]['marker']['symbol'] == 'circle'
