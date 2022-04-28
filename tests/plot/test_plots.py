import pytest

import numpy as np
import pandas as pd

from endaq import plot


@pytest.fixture(params=["float", "datetime"])
def generate_dataframe(request):
    freqs = np.arange(10)
    times = np.zeros_like(freqs)

    df = pd.DataFrame({
        'frequency (Hz)': np.concatenate((freqs, freqs, freqs)),
        'timestamp': np.concatenate((times, times + 30, times + 60)),
        'value': np.random.random(30)
    })
    df['variable'] = 'X'

    if request.param == 'datetime':
        df.timestamp = pd.to_datetime(df.timestamp, unit='D', utc=True)

    return df


class TestSpectrumOverTime:

    def test_heatmap(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Heatmap')
        assert fig['data'][0]['type'] == 'heatmap'

    def test_surface(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Surface')
        assert fig['data'][0]['type'] == 'surface'

    def test_waterfall(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Waterfall')
        assert fig['data'][0]['type'] == 'scatter3d'

    def test_animation(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Animation')
        assert len(fig['data']) == 4

    def test_lines(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Lines')
        assert len(fig['data']) == 9

    def test_peak(self, generate_dataframe):
        fig = plot.spectrum_over_time(generate_dataframe, 'Peak')
        assert fig['data'][0]['marker']['symbol'] == 'circle'


def test_octave_spectrogram():
    time = np.arange(1000) / 1000
    df_accel = pd.DataFrame({
        'time': np.concatenate((time, time + 1.0, time + 2.0)),
        'A': np.random.random(3000)
    }).set_index('time')

    df, fig = plot.octave_spectrogram(df_accel, window=0.1, max_freq=50)
    assert fig['data'][0]['type'] == 'heatmap'

