# endaq-plot - Painless Plotting Of Sensor Data

endaq-plot is a module comprising a collection of plotting utilities for sensor data analysis. It leverages Plotly in order to produce interactive plots, and makes creating powerful visualizations simple and easy for those new to Python.

endaq-plot is a sub-module of the larger enDAQ ecosystem. See [the endaq package](https://github.com/MideTechnology/endaq-python) for more details.

## Usage Examples

For these examples we assume there is a Pandas DataFrame named `df` which has it's index as time stamps and it's one column being sensor values (e.g. x-axis accleration, or pressure).   It also assumes there is a Pandas DataFrame `attribute_df` which contains all the attribute data about various data files.  More information can be found about how to get this data from enDAQ IDE files in the [endaq-cloud readme](https://github.com/MideTechnology/endaq-python/tree/main/endaq/cloud).

```python
from endaq.plot import octave_spectrogram, multi_file_plot_attributes, octave_psd_bar_plot
from endaq.plot.utilities import set_theme
```

### Setting The Aesthetic Theme

```python
set_theme(theme='endaq')
```

### Creating Spectrograms With Octave Spaced Frequencies

```python
data_df, fig = octave_spectrogram(df, window=.15)
fig.show()
```

![Spectrogram With Octave Spaced Frequencies](https://i.imgur.com/929aszu.png)

### Creating PSD Bar Plots With Octave Spaced Frequencies

```python
fig = octave_psd_bar_plot(df, yaxis_title="Magnitude")
fig.show()
```

![PSD Bar Plot With Octave Spaced Frequencies](https://i.imgur.com/ueqcVTQ.png)

### Plot Attributes In Figure With Subplots

```Python
fig = multi_file_plot_attributes(attribute_df)
fig.show()
```

![Attributes Plotted As Subplots](https://i.imgur.com/5Yy4DN7.png)

## Other Links
- the endaq package - https://github.com/MideTechnology/endaq-python
- the enDAQ homepage - https://endaq.com/
