from __future__ import annotations

import typing
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import colors
import scipy.spatial.transform
from plotly.subplots import make_subplots
from scipy import signal, spatial
from typing import Optional
from collections.abc import Container
import datetime

from endaq.calc import sample_spacing
from endaq.calc.psd import to_octave, welch, rolling_psd

from .utilities import determine_plotly_map_zoom, get_center_of_coordinates
from .dashboards import rolling_enveloped_dashboard

__all__ = [
    'multi_file_plot_attributes',
    'general_get_correlation_figure',
    'get_pure_numpy_2d_pca',
    'gen_map',
    'octave_spectrogram',
    'octave_psd_bar_plot',
    'rolling_min_max_envelope',
    'around_peak',
    'animate_quaternion',
    'spectrum_over_time'
]

DEFAULT_ATTRIBUTES_TO_PLOT_INDIVIDUALLY = np.array([
    'accelerationPeakFull', 'accelerationRMSFull', 'velocityRMSFull', 'psuedoVelocityPeakFull',
    'displacementRMSFull', 'gpsSpeedFull', 'gyroscopeRMSFull', 'microphonoeRMSFull',
    'temperatureMeanFull', 'pressureMeanFull'])


def multi_file_plot_attributes(multi_file_db: pd.DataFrame,
                               attribs_to_plot: np.ndarray = DEFAULT_ATTRIBUTES_TO_PLOT_INDIVIDUALLY,
                               recording_colors: Optional[Container] = None,
                               width_per_subplot: int = 400) -> go.Figure:
    """
    Creates a Plotly figure plotting all the desired attributes from the given DataFrame.

    :param multi_file_db: The Pandas DataFrame of data to plot attributes from
    :param attribs_to_plot: A numpy ndarray of strings with the names of the attributes to plot
    :param recording_colors: The colors to make each of the points (All will be the same color if None is given)
    :param width_per_subplot: The width to make every subplot
    :return: A Plotly figure of all the subplots desired to be plotted
    """
    if not isinstance(attribs_to_plot, np.ndarray):
        raise TypeError(
            "Instead of an ndarray given for 'attribs_to_plot', a variable of type %s was given" % str(type(attribs_to_plot)))

    if len(attribs_to_plot) == 0:
        raise ValueError("At least one value must be given for 'attribs_to_plot'")

    if not all(isinstance(row, str) for row in attribs_to_plot):
        raise TypeError("All rows to plot must be given as a string!")

    should_plot_row = np.array([not multi_file_db[r].isnull().all() for r in attribs_to_plot])

    attribs_to_plot = attribs_to_plot[should_plot_row]

    fig = make_subplots(
        rows=1,
        cols=len(attribs_to_plot),
        subplot_titles=attribs_to_plot,
    )

    for j, row_name in enumerate(attribs_to_plot):
        fig.add_trace(
            go.Scatter(
                x=multi_file_db['recording_ts'],
                y=multi_file_db[row_name],
                name=row_name,
                mode='markers',
                text=multi_file_db['serial_number_id'].values,
                marker_color=recording_colors,
            ),
            row=1,
            col=j + 1,
        )

    return fig.update_layout(width=len(attribs_to_plot)*width_per_subplot, showlegend=False)


def general_get_correlation_figure(merged_df: pd.DataFrame,
                                   color_col: str = 'serial_number_id',
                                   color_discrete_map: Optional[dict] = None,
                                   hover_names: Optional[Container] = None,
                                   characteristics_to_show_on_hover: list = [],
                                   starting_cols: Container = None) -> go.Figure:
    """
    A function to create a plot with two drop-down menus, each populated with a set of options corresponding to the
    scalar quantities contained in the given dataframe.   The data points will then be plotted with the X and Y axis
    corresponding to the selected attributes from the drop-down menu.

    :param merged_df: A Pandas DataFrame of data to use for producing the plot
    :param color_col: The column name in the given dataframe (as ``merged_df``) that is used to color
     data points with.   This is used in combination with the color_discrete_map parameter
    :param color_discrete_map: A dictionary which maps the values given to color data points based on (see the
     color_col parameter description) to the colors that these data points should be
    :param hover_names: The names of the points to display when they are hovered on
    :param characteristics_to_show_on_hover: The set of characteristics of the data to display when hovered over
    :param starting_cols: The two starting columns for the dropdown menus (will be the first two available
     if None is given)
    :return: The interactive Plotly figure
    """
    cols = [col for col, t in zip(merged_df.columns, merged_df.dtypes) if t != np.object]

    point_naming_characteristic = merged_df.index if hover_names is None else hover_names

    # This is not necessary, but usually produces easily discernible correlations or groupings of files/devices.
    # The hope is that when the initial plot has these characteristics, it will encourage
    # the exploration of this interactive plot.
    start_dropdown_indices = [0, 1]
    first_x_var = cols[0]
    first_y_var = cols[1]
    if starting_cols is not None and starting_cols[0] in cols and starting_cols[1] in cols:
        for j, col_name in enumerate(cols):
            if col_name == starting_cols[0]:
                first_x_var, start_dropdown_indices[0] = col_name, j
            if col_name == starting_cols[1]:
                first_y_var, start_dropdown_indices[1] = col_name, j

    # Create the scatter plot of the initially selected variables
    fig = px.scatter(
        merged_df,
        x=first_x_var,
        y=first_y_var,
        color=color_col,
        color_discrete_map=color_discrete_map,
        hover_name=point_naming_characteristic,
        hover_data=characteristics_to_show_on_hover,
        # width=800,
        # height=500,
    )

    # Create the drop-down menus which will be used to choose the desired file characteristics for comparison
    drop_downs = []
    for axis in ['x', 'y']:
        drop_downs.append([
            dict(
                method="update",
                args=[{axis: [merged_df[cols[k]]]},
                      {'%saxis.title.text' % axis: cols[k]},
                      # {'color': recording_colors},
                      ],
                label=cols[k]) for k in range(len(cols))
        ])

    # Sets up various apsects of the Plotly figure that is currently being produced.  This ranges from
    # aethetic things, to setting the dropdown menues as part of the figure
    fig.update_layout(
        title_x=0.4,
        # width=800,
        # height=500,
        showlegend=False,
        updatemenus=[{
            'active': start_j,
            'buttons': drop_down,
            'x': 1.125,
            'y': y_height,
            'xanchor': 'left',
            'yanchor': 'top',
        } for drop_down, start_j, y_height in zip(drop_downs, start_dropdown_indices, [1, .85])])

    # Adds text labels for the two drop-down menus
    for axis, height in zip(['X', 'Y'], [1.05, .9]):
        fig.add_annotation(
            x=1.1,  # 1.185,
            y=height,
            xref='paper',
            yref='paper',
            showarrow=False,
            xanchor='left',
            yanchor='top',
            text="%s-Axis Measurement" % axis,
        )
    return fig


def get_pure_numpy_2d_pca(df: pd.DataFrame,
                          color_col: str = 'serial_number_id',
                          color_discrete_map: Optional[dict] = None,
                          ) -> go.Figure:
    """
    Get a Plotly figure of the 2D PCA for the given DataFrame.   This will have dropdown menus to select
    which components are being used for the X and Y axis.

    :param df: The dataframe of points to compute the PCA with
    :param color_col: The column name in the given dataframe (as ``df``) that is used to color
     data points with.   This is used in combination with the color_discrete_map parameter
    :param color_discrete_map: A dictionary which maps the values given to color data points based on (see the
     color_col parameter description) to the colors that these data points should be
    :return: A plotly figure as described in the main function description

    .. todo::
     - Add type checking statements to ensure the given dataframe contains enough values of the desired type
     - Add type checking statements to ensure the recording_colors given (if not None) are the proper length
    """

    # Drop all non-float64 type columns, and drop all columns with standard deviation of 0 because this will result
    # in division by 0
    X = df.loc[:, (df.std() != 0) & (np.float64 == df.dtypes)].dropna(axis='columns')

    # Standardize the values (so that mean of each variable is 0 and standard deviation is 1)
    X = (X - X.mean()) / X.std()

    # Get the shape of the data to compute PCA for
    n, m = X.shape

    # Compute covariance matrix
    covariance = np.dot(X.T, X) / (n - 1)

    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(covariance)

    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)

    # Create a new DataFrame with column names describing the fact that the values are the principal components
    pca_df = pd.DataFrame(X_pca)
    pca_df.columns = (pca_df.columns + 1).map(lambda x: "Component %d" % x)

    # Produce the Plotly figure for the points after being transfored with the PCA computed,
    # with dropdown menus to allow selection of what PCA components are being analyzed
    fig = general_get_correlation_figure(
        pca_df,
        color_col=df[color_col],
        color_discrete_map=color_discrete_map,
        characteristics_to_show_on_hover=[df.index],
        hover_names=df.index
    )

    return fig


def gen_map(df_map: pd.DataFrame, mapbox_access_token: str, filter_points_by_positive_groud_speed: bool = True,
            color_by_column: str = "GNSS Speed: Ground Speed") -> go.Figure:
    """
    Plots GPS data on a map from a single recording, shading the points based some characteristic
    (defaults to ground speed).
    
    :param df_map: The pandas dataframe containing the recording data.
    :param mapbox_access_token: The access token (or API key) needed to be able to plot against
     a map.
    :param filter_points_by_positive_groud_speed: A boolean variable, which will filter
     which points are plotted by if they have corresponding positive ground speeds.  This helps
     remove points which didn't actually have a GPS location found (was created by a bug in the hardware I believe).
    :param color_by_column: The dataframe column title to color the plotted points by.
    """
    if filter_points_by_positive_groud_speed:
        df_map = df_map[df_map["GNSS Speed: Ground Speed"] > 0]
    
    zoom = determine_plotly_map_zoom(lats=df_map["Location: Latitude"], lons=df_map["Location: Longitude"])
    center = get_center_of_coordinates(lats=df_map["Location: Latitude"], lons=df_map["Location: Longitude"])
    
    px.set_mapbox_access_token(mapbox_access_token)
    
    fig = px.scatter_mapbox(
        df_map,
        lat="Location: Latitude",
        lon="Location: Longitude",
        color=color_by_column,
        size_max=15,
        zoom=zoom - 1,
        center=center,
    )

    return fig
    

def octave_spectrogram(df: pd.DataFrame, window: float, bins_per_octave: int = 3, freq_start: float = 20.0,
                       max_freq: float = float('inf'), db_scale: bool = True, log_scale_y_axis: bool = True
                       ) -> tuple[pd.DataFrame, go.Figure]:
    """
    Produces an octave spectrogram of the given data, this is a wrapper around 
    :py:func:`~endaq.calc.psd.rolling_psd()` and :py:func:`~endaq.plot.spectrum_over_time()`
    
    :param df: The dataframe of sensor data.  This must only have 1 column.
    :param window: The time window for each of the columns in the spectrogram
    :param bins_per_octave: The number of frequency bins per octave
    :param freq_start: The center of the first frequency bin
    :param max_freq: The maximum frequency to plot
    :param db_scale: If the spectrogram should be log-scaled for visibility (with 10*log10(x))
    :param log_scale_y_axis: If the y-axis of the plot should be log scaled
    :return: a tuple containing:
        - dataframe of the spectrogram data
        - the corresponding plotly figure
    """
    if len(df.columns) != 1:
        raise ValueError("The parameter 'df' must have only one column of data!")

    num_slices = int(len(df) * sample_spacing(df) / window)

    df_psd = rolling_psd(df, num_slices=num_slices, octave_bins=bins_per_octave,
                         fstart=freq_start)

    return df_psd, spectrum_over_time(df_psd, freq_max=max_freq, log_val=db_scale, log_freq=log_scale_y_axis)
    

def octave_psd_bar_plot(df: pd.DataFrame, bins_per_octave: int = 3, f_start: float = 20.0, yaxis_title: str = '',
                        log_scale_y_axis: bool = True) -> go.Figure:
    """
    Produces a bar plot of an octave psd.

    :param df: The dataframe of sensor data
    :param bins_per_octave: The number of frequency bins per octave
    :param f_start: The center of the first frequency bin
    :param yaxis_title: The text to label the y-axis
    :param log_scale_y_axis: If the y-axis should be log scaled
    """
    psd_df = welch(df, 1, scaling='spectrum')

    octave_psd_df = to_octave(
        psd_df,
        f_start, 
        bins_per_octave,
        agg='sum',
    )

    bar_widths = octave_psd_df.index * (2 ** (.5/bins_per_octave) - 2 ** (-.5/bins_per_octave))

    bar = go.Bar(
        x=octave_psd_df.index.values,
        y=octave_psd_df.values.squeeze(),
        width=bar_widths,
    )
    layout = go.Layout(
        yaxis={'title': yaxis_title},
        xaxis={'title': 'Frequency (Hz)'},
    )

    fig = go.Figure(data=bar, layout=layout)

    if log_scale_y_axis:
        fig.update_xaxes(type="log")

    return fig


def rolling_min_max_envelope(df: pd.DataFrame, desired_num_points: int = 250, plot_as_bars: bool = False,
                             plot_title: str = "", opacity: float = 1.0,
                             colors_to_use: Optional[Container] = None) -> go.Figure:
    """
    A function to create a Plotly Figure to plot the data for each of the available data sub-channels, designed to
    reduce the number of points/data being plotted without minimizing the insight available from the plots.  It will
    plot either an envelope for rolling windows of the data (plotting the max and the min as line plots), or a bar based
    plot where the top of the bar (rectangle) is the highest value in the time window that bar spans, and the bottom of
    the bar is the lowest point in that time window (choosing between them is done with the `plot_as_bars` parameter).

    :param df: The dataframe of sub-channel data indexed by time stamps
    :param desired_num_points: The desired number of points to be plotted for each subchannel.  The number of points
     will be reduced from it's original sampling rate by applying metrics (e.g. min, max) over sliding windows
     and then using that information to represent/visualize the data contained within the original data.  If less than
     the desired number of points are present, then a sliding window will NOT be used, and instead the points will be
     plotted as they were originally recorded  (also the subchannel will NOT be plotted as a bar based plot even if
     `plot_as_bars` was set to true).
    :param plot_as_bars: A boolean value indicating if the data should be visualized as a set of rectangles, where a
     shaded rectangle is used to represent the maximum and minimum values of the data during the time window
     covered by the rectangle.  These maximum and minimum values are visualized by the locations of the top and bottom
     edges of the rectangle respectively, unless the height of the rectangle would be 0, in which case a line segment
     will be displayed in it's place.  If this parameter is `False`, two lines will be plotted for each
     of the sub-channels in the figure being created, creating an 'envelope' around the data.  An 'envelope' around the
     data consists of a line plotted for the maximum values contained in each of the time windows, and another line
     plotted for the minimum values.  Together these lines create a boundary which contains all the data points
     recorded in the originally recorded data.
    :param plot_title: The title for the plot
    :param opacity: The opacity to use for plotting bars/lines
    :param colors_to_use: An "array-like" object of strings containing colors to be cycled through for the sub-channels.
     If `None` is given (which is the default), then the `colorway` variable in Plotly's current theme/template will
     be used to color the data on each of the sub-channels uniquely, repeating from the start of the `colorway` if
     all colors have been used.
    :return: The Plotly Figure with the data plotted
    """

    return rolling_enveloped_dashboard(
        {plot_title: df},
        desired_num_points=desired_num_points,
        subplot_colors=colors_to_use,
        plot_as_bars=plot_as_bars,
        plot_full_single_channel=True,
        opacity=opacity
    )


def around_peak(df: pd.DataFrame, num: int = 1000, leading_ratio: float = 0.5):
    """
    A function to plot the data surrounding the largest peak (or valley) in the given data.
    The "peak" is defined by the point in the absolute value of the given data with the largest value.

    :param df: A dataframe indexed by time stamps
    :param num: The number of points to plot
    :param leading_ratio: The ratio of the data to be viewed that will come before the peak
    :return: A Plotly figure containing the plot
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The `df` parmeter must be of type `pd.DataFrame` but was given type {type(df)} instead.")

    if not isinstance(num, int):
        raise TypeError(f"The `num` parameter must be an `int` type, but was given {type(num)}.")

    if not isinstance(leading_ratio, float):
        raise TypeError(f"The `leading_ratio` parameter must be a `float` type, but was given {type(leading_ratio)}.")

    if len(df) == 0:
        raise ValueError(f"The parameter `df` must have nonzero length, but has shape {df.shape} instead")

    if num < 3:
        raise ValueError(f"The `num` parameter must be at least 3, but {num} was given.")

    if leading_ratio < 0 or leading_ratio > 1:
        raise ValueError("The `leading_ratio` parameter must be a float value in the "
                         f"range [0,1], but was given {leading_ratio} instead.")

    max_i = df.abs().max(axis=1).reset_index(drop=True).idxmax()

    # These can go below and above the number of valid indices, but that can be ignored since
    # they'll only be used to slice the data in a way that is okay to go over/under
    window_start = max(0, max_i - int(num * leading_ratio))
    window_end = min(len(df), max_i + int(num * (1-leading_ratio)))

    return px.line(df.iloc[window_start: window_end])


def animate_quaternion(df, rate=6., scale=1.):
    """
    A function to animate orientation as a set of axis markers derived from quaternion data.

    :param df: A dataframe of quaternion data indexed by seconds.
               Columns must be 'X', 'Y', 'Z', and 'W'.  The order of the columns does not matter.
    :param rate: The number of frames per second to animate.
    :param scale: Time-scaling to adjust how quickly data is parsed.  Higher speeds will make the
                  file run faster, lower speeds will make the file run slower.
    :return: A Plotly figure containing the plot
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The `df` parmeter must be of type `pd.DataFrame` but was given type {type(df)} instead.")
    elif len(df) == 0:
        raise ValueError(f"The parameter `df` must have nonzero length, but has shape {df.shape} instead")

    if not isinstance(rate, (int, float)):
        raise TypeError(f"The `rate` parameter must be an `int` or `float` type, but was given {type(rate)}.")
    elif rate <= 0:
        raise ValueError(f"The `rate` parameter must be a positive number, but was {rate}")

    if not isinstance(scale, (int, float)):
        raise TypeError(f"The `scale` parameter must be an `int` or `float` type, but was given {type(scale)}.")
    elif scale <= 0:
        raise ValueError(f"The `scale` parameter must be a positive number, but was {scale}")

    start = df.index[0]
    end = df.index[-1]

    fs = rate/scale

    t = np.arange(np.ceil((end - start)*fs))/fs + start

    r = spatial.transform.Rotation.from_quat(df[['X', 'Y', 'Z', 'W']])

    r_resample = spatial.transform.Slerp(df.index, r)(t)

    vals = []
    for _t, x, y, z in zip(
            t,
            r_resample.apply(([1, 0, 0])),
            r_resample.apply(([0, 1, 0])),
            r_resample.apply(([0, 0, 1])),
            ):

        vals.append({'time': _t, 'x': 0,    'y': 0,    'z': 0,    'channel': 'X'})
        vals.append({'time': _t, 'x': x[0], 'y': x[1], 'z': x[2], 'channel': 'X'})
        vals.append({'time': _t, 'x': 0,    'y': 0,    'z': 0,    'channel': 'Y'})
        vals.append({'time': _t, 'x': y[0], 'y': y[1], 'z': y[2], 'channel': 'Y'})
        vals.append({'time': _t, 'x': 0,    'y': 0,    'z': 0,    'channel': 'Z'})
        vals.append({'time': _t, 'x': z[0], 'y': z[1], 'z': z[2], 'channel': 'Z'})

    df_axes = pd.DataFrame(vals)

    fig = px.line_3d(
            df_axes,
            x='x',
            y='y',
            z='z',
            color='channel',
            animation_frame='time',
            animation_group='channel',
            ).update_layout(
            font_size=16,
            legend_title_text='',
            title_text='Quaternion 3D Animation',
            title_x=0.5,
            title_font_size=24,
            scene=dict(
                    xaxis=dict(range=[-1.1, 1.1], nticks=5),
                    yaxis=dict(range=[-1.1, 1.1], nticks=5),
                    zaxis=dict(range=[-1.1, 1.1], nticks=5),
                    aspectmode='cube',
                    ),
            )
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = int(1000//rate)

    return fig


def spectrum_over_time(
        df: pd.DataFrame,
        plot_type: typing.Literal["Heatmap",
                                  "Surface",
                                  "Waterfall",
                                  "Animation",
                                  "Peak",
                                  "Lines"] = "Heatmap",
        var_column: str = 'variable',
        var_to_process: str = None,
        time_column: str = 'timestamp',
        freq_column: str = 'frequency (Hz)',
        val_column: str = 'value',
        freq_min: float = None,
        freq_max: float = None,
        log_freq: bool = False,
        log_val: bool = False,
        round_time: bool = True,
        waterfall_line_sequence: bool = True,
        waterfall_line_color: str = '#EE7F27',
        min_median_max_line_color: str = '#6914F0',
) -> go.Figure:
    """
    Generate a 3D Plotly figure from a stacked spectrum to visualize how the frequency content changes over time
    
    :param df: the input dataframe with columns defining the frequency content, timestamps, values, and variables,
        see the following functions which provides outputs that would then be passed into this function as an input:
        * :py:func:`~endaq.calc.fft.rolling_fft()`
        * :py:func:`~endaq.calc.psd.rolling_psd()`
        * :py:func:`~endaq.calc.shock.rolling_shock_spectrum()`
        * :py:func:`~endaq.batch.GetDataBuilder.add_psd()`
        * :py:func:`~endaq.batch.GetDataBuilder.add_pvss()`
    :param plot_type: the type of plot to display the spectrum, options are:
        * `Heatmap`:  a 2D visualization with the color defining the z value (the default)
        * `Surface`: similar to `Heatmap` but the z value is also projected "off the page"
        * `Waterfall`: distinct lines are plotted per time slice in a 3D view
        * `Animation`: a 2D display of the waterfall but the spectrum changes with animation frames
        * `Peak`: per timestamp the peak frequency is determined and plotted against time
        * `Lines`: the value in each frequency bin is plotted against time
    :param var_column: the column name in the dataframe that defines the different variables, default is `"variable"`
    :param var_to_process: the variable value in the `var_column` to filter the input df down to,
        if none is provided (the default) this function will filter to the first value
    :param time_column: the column name in the dataframe that defines the timestamps, default is `"timestamp"`
    :param freq_column: the column name in the dataframe that defines the frequency, default is `"frequency (Hz)"`
    :param val_column: the column name in the dataframe that defines the values, default is `"value"`
    :param freq_min: the minimum of the y axis (frequency) to include in the figure,
        default is None meaning it will display all content
    :param freq_max: the maximum of the y axis (frequency) to include in the figure,
        default is None meaning it will display all content
    :param log_freq: if `True` the frequency will be in a log scale, default is `False`
    :param log_val: if `True` the values will be in a log scale, default is `False`
    :param round_time: if `True` (default) the time values will be rounded to the nearest second for datetimes and
        hundredths of a second for floats
    :param waterfall_line_sequence: if `True` the waterfall line colors are defined with a color scale,
        if `False` all lines will have the same color, default is `True`
    :param waterfall_line_color: the color to use for all lines in the Waterfall plot if `waterfall_line_sequence` is
        `False`
    :param min_median_max_line_color: the color to use for the min, max, and median lines in the Animation, if set to
        `None` these lines won't be added, default is `'#6914F0'`
    :return: a Plotly figure visualizing the spectrum over time


    Here's a few examples from a dataset recorded with an enDAQ sensor on a motorcycle
    as it revved the engine which resulted in changing frequency content
    
    .. code:: python

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        #Get the Data
        df_vibe = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-vibration-moving-frequency.csv',index_col=0)
        df_vibe = df_vibe - df_vibe.median()

        #Calculate a Rolling FFT
        fft = endaq.calc.fft.rolling_fft(df_vibe, num_slices=200, add_resultant=True)

        #Visualize the Rolling FFT as a Heatmap
        heatmap = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Heatmap', freq_max=200, var_to_process='Resultant')
        heatmap.show()

        #Visualize as the peak frequency
        peak = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Peak', freq_max=200, var_to_process='Resultant')
        peak.show()

        #Visualize as a Surface Plot
        surface = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Surface', freq_max=200, var_to_process='Resultant')
        surface.show()

        #Visualize as a Waterfall
        waterfall = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Waterfall', freq_max=200, var_to_process='Resultant')
        waterfall.show()

        #Visualize as an Animation
        animation = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Animation', log_val=True, freq_max=200, var_to_process='Resultant')
        animation.show()

        #Calculate a Rolling PSD with Units as RMS**2
        psd = endaq.calc.psd.rolling_psd(df_vibe, num_slices=200, scaling='parseval', add_resultant=True, octave_bins=1, fstart=1, agg='sum')
        psd['value'] = psd['value'] ** 0.5

        #Visualize the energy in each frequency bin
        lines = endaq.plot.plots.spectrum_over_time(psd, plot_type = 'Lines',  log_val=False, var_to_process='Resultant')
        lines.show()

    .. plotly::
        :fig-vars: heatmap, peak, surface, waterfall, animation, lines

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        #Get the Data
        df_vibe = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-vibration-moving-frequency.csv',index_col=0)
        df_vibe = df_vibe - df_vibe.median()

        #Calculate a Rolling FFT
        fft = endaq.calc.fft.rolling_fft(df_vibe, num_slices=200, add_resultant=True)

        #Visualize the Rolling FFT as a Heatmap
        heatmap = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Heatmap', freq_max=200, var_to_process='Resultant')
        heatmap.show()

        #Visualize as the peak frequency
        peak = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Peak', freq_max=200, var_to_process='Resultant')
        peak.show()

        #Visualize as a Surface Plot
        surface = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Surface', freq_max=200, var_to_process='Resultant')
        surface.show()

        #Visualize as a Waterfall
        waterfall = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Waterfall', freq_max=200, var_to_process='Resultant')
        waterfall.show()

        #Visualize as an Animation
        animation = endaq.plot.plots.spectrum_over_time(fft, plot_type = 'Animation', log_val=True, freq_max=200, var_to_process='Resultant')
        animation.show()

        #Calculate a Rolling PSD with Units as RMS**2
        psd = endaq.calc.psd.rolling_psd(df_vibe, num_slices=200, scaling='parseval', add_resultant=True, octave_bins=1, fstart=1, agg='sum')
        psd['value'] = psd['value'] ** 0.5

        #Visualize the energy in each frequency bin
        lines = endaq.plot.plots.spectrum_over_time(psd, plot_type = 'Lines',  log_val=False, var_to_process='Resultant')
        lines.show()

    """
    df = df.copy()

    # Filter to one variable
    if var_to_process is None:
        var_to_process = df[var_column].unique()[0]
    df = df.loc[df[var_column] == var_to_process]

    # Round time
    if round_time:
        if isinstance(df[time_column].iloc[0], datetime.datetime):
            df[time_column] = df[time_column].round('s')
        else:
            df[time_column] = np.round(df[time_column].to_numpy(), 2)

    # Filter frequency
    df = df.loc[df[freq_column] > 0.0]
    if freq_max is not None:
        df = df.loc[df[freq_column] < freq_max]
    if freq_min is not None:
        df = df.loc[df[freq_column] > freq_min]

    # Remove 0s
    df = df.loc[df[val_column] > 0]

    # Check Length of Dataframe
    if len(df) > 50000:
        warnings.warn(
            "plot data is very large, may be unresponsive, suggest limiting frequency range and/or using less slices",
            RuntimeWarning,
        )

    # Create pivot table
    df_pivot = df.copy()
    first_step = df_pivot[freq_column].iloc[1] - df_pivot[freq_column].iloc[0]
    second_step = df_pivot[freq_column].iloc[2] - df_pivot[freq_column].iloc[1]
    if first_step == second_step:
        round_freq = np.round(df_pivot[freq_column].min(), 0)
        df_pivot[freq_column] = np.round(df_pivot[freq_column].to_numpy() / round_freq, 0) * round_freq
    df_pivot = df_pivot.pivot(columns=time_column, index=freq_column, values=val_column)

    # Heatmap & Surface
    if plot_type in ["Heatmap", "Surface"]:
        # Deal with Datetime
        x_type = float
        if isinstance(df_pivot.columns[0], datetime.datetime):
            if plot_type == "Surface":
                df_pivot.columns = (df_pivot.columns - df_pivot.columns[0]).total_seconds()
            else:
                x_type = str

        # Build Dictionary of Plot Data, Apply Log Scale If Needed
        data_dict = {
            'x': df_pivot.columns.astype(x_type),
            'y': df_pivot.index.astype(float),
            'z': df_pivot.to_numpy().astype(float),
            'connectgaps': True
        }
        if log_val:
            data_dict['z'] = np.log10(data_dict['z'])

        # Generate Figures
        if plot_type == "Heatmap":
            fig = go.Figure(data=go.Heatmap(data_dict, zsmooth='best')).update_layout(
                xaxis_title_text='Timestamp', yaxis_title_text='Frequency (Hz)')
            if log_freq:
                fig.update_layout(yaxis_type='log')
        else:
            fig = go.Figure(data=go.Surface(data_dict))
        fig.update_traces(showscale=False)

    # Waterfall
    elif plot_type == 'Waterfall':
        # Define Colors
        if waterfall_line_sequence:
            color_sequence = colors.sample_colorscale(
                [[0.0, '#6914F0'],
                 [0.2, '#3764FF'],
                 [0.4, '#2DB473'],
                 [0.6, '#FAC85F'],
                 [0.8, '#EE7F27'],
                 [1.0, '#D72D2D']], len(df[time_column].unique()))
        else:
            color_sequence = [waterfall_line_color]

        # Deal with Datetime
        df['label_column'] = df[time_column]
        if isinstance(df[time_column].iloc[0], datetime.datetime):
            df[time_column] = (df[time_column] - df[time_column].iloc[0]).dt.total_seconds()
            df['label_column'] = df['label_column'].dt.tz_convert(None).astype(str)

        # Generate Figure
        fig = px.line_3d(
            df,
            x=time_column,
            y=freq_column,
            z=val_column,
            color='label_column',
            color_discrete_sequence=color_sequence).update_layout(
            legend_orientation='v',
            legend_y=1,
            legend_title_text='Timestamps'
        )

    # Animation
    elif plot_type == 'Animation':
        # Deal with Datetime
        if isinstance(df[time_column].iloc[0], datetime.datetime):
            df[time_column] = df[time_column].dt.tz_convert(None).astype(str)

        # Generate Figure
        fig = px.line(
            df,
            animation_frame=time_column,
            x=freq_column,
            y=val_column,
            log_y=log_val,
            log_x=log_freq
        ).update_layout(
            showlegend=False,
            yaxis_title_text='',
            xaxis_title_text='Frequency (Hz)'
        )

        # Add min, max, median lines
        if min_median_max_line_color is not None:
            # Add Max
            df_t = df_pivot.max(axis=1).dropna()
            fig.add_trace(go.Scattergl(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
            ))

            # Add Min
            df_t = df_pivot.min(axis=1).dropna()
            fig.add_trace(go.Scattergl(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
            ))

            # Add Median
            df_t = df_pivot.median(axis=1).dropna()
            fig.add_trace(go.Scattergl(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
                line_dash='dash',
            ))

    # Peak Frequencies
    elif plot_type == 'Peak':
        fig = px.line(
            df_pivot.idxmax(),
            log_y=log_freq
        ).update_layout(
            showlegend=False,
            yaxis_title_text='Peak Frequency (Hz)',
            xaxis_title_text='Timestamp'
        )

    # Lines per Frequency Bin
    elif plot_type == 'Lines':
        df_pivot.index = np.round(df_pivot.index, 2)
        fig = px.line(
            df_pivot.T,
            log_y=log_val
        ).update_layout(
            legend_title_text='Frequency Bin (Hz)',
            yaxis_title_text='',
            xaxis_title_text='Timestamp',
            legend_orientation='v',
            legend_y=1,
        )

    else:
        raise ValueError(f"invalid plot type {plot_type}")

    # Add Labels to 3D plots
    if plot_type in ["Surface", "Waterfall"]:
        fig.update_scenes(
            aspectratio_x=2.0,
            aspectratio_y=1.0,
            aspectratio_z=0.3,
            xaxis_title_text='Timestamp',
            yaxis_title_text='Frequency (Hz)',
            zaxis_title_text=''
        )
        if log_freq:
            fig.update_scenes(yaxis_type='log')
        if log_val:
            fig.update_scenes(zaxis_type='log')
        fig.update_layout(scene_camera=dict(eye=dict(x=-1.5, y=-1.5, z=1)))

    return fig
