import pytest
from datetime import datetime, timedelta

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

@pytest.fixture(params=["float", "datetime"])
def generate_time_dataframe(request):
    time = np.arange(1000) / 1000
    df_accel = pd.DataFrame({
        'time': np.concatenate((time, time + 1.0, time + 2.0)),
        'A': np.random.random(3000),
        'B': np.random.random(3000)
    }).set_index('time')

    if request.param == 'datetime':
        start_date = datetime.strptime('2021-03-21 00:00', '%Y-%m-%d %H:%M')
        df_accel.index = [start_date + timedelta(seconds=diff) for diff in df_accel.index]

    return df_accel


def test_octave_spectrogram(generate_time_dataframe):
    df, fig = plot.octave_spectrogram(generate_time_dataframe[['A']], window=0.1, max_freq=50)
    assert fig['data'][0]['type'] == 'heatmap'


def test_around_peak(generate_time_dataframe):
    fig = plot.around_peak(generate_time_dataframe)
    assert fig['data'][0]['type'] == 'scattergl'


class TestRollingMinMaxEnvelope:

    def test_lines(self, generate_time_dataframe):
        fig = plot.rolling_min_max_envelope(generate_time_dataframe)
        assert fig['data'][0]['type'] == 'scatter'

    def test_lines(self, generate_time_dataframe):
        fig = plot.rolling_min_max_envelope(generate_time_dataframe, plot_as_bars=True)
        assert fig['data'][0]['type'] == 'bar'
