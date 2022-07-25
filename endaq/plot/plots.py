from __future__ import annotations

from collections.abc import Container
import datetime
import typing
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import colors
from plotly.subplots import make_subplots
from scipy import spatial
import idelib.dataset

from endaq.calc import utils, psd
from endaq.ide import get_channel_table
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
    'spectrum_over_time',
    'pvss_on_4cp',
    'table_plot',
    'table_plot_from_ide'
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

    # Sets up various aspects of the Plotly figure that is currently being produced.  This ranges from
    # aesthetic things, to setting the dropdown menus as part of the figure
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


def gen_map(df_map: pd.DataFrame,
            mapbox_access_token: str = None,
            lat: str = "Latitude",
            lon: str = "Longitude",
            color_by_column: str = None,
            filter_positive_color_vals: bool = False,
            hover_data: list[str] = [],
            size_max: float = 15.0,
            zoom_offset: float = -2.0
            ) -> go.Figure:
    """
    Plots GPS data on a map from a single recording, shading the points based on one of several characteristics
        (defaults to ground speed).
    
    :param df_map: The pandas dataframe containing the recording data.
    :param mapbox_access_token: The access token (or API key) needed to be able to plot against a map using Mapbox,
        `create a free account here <https://www.mapbox.com/pricing>`_

        * If no access token is provided, a `"stamen-terrain"` tile will be used,
            `see Plotly for more information <https://plotly.com/python/mapbox-layers/>`_
    :param lat: The dataframe column title to use for latitude
    :param lon: The dataframe column title to use for longitude
    :param color_by_column: The dataframe column title to color the plotted points by.
        If `None` is provided (the default), all the points will be the same color
    :param filter_positive_color_vals: A boolean variable, which will filter which points are plotted by
        if they have corresponding positive values, default is `False`
    :param hover_data: The list of dataframe column titles with data to display when the mouse hovers over a datapoint
    :param size_max: The size of the scatter points in the map, default 15
    :param zoom_offset: The offset to apply to the zoom, default -2, this is influenced by the final figure size

    Here is an example map of a drive from Boston Logan Airport to Mide Technology

    .. code:: python

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        # Get GPS Data
        gps = pd.read_csv('https://info.endaq.com/hubfs/data/mide-map-gps-data.csv')

        # Generate & Show Map
        fig = endaq.plot.gen_map(
            gps,
            lat="Latitude",
            lon="Longitude",
            color_by_column="Ground Speed",
            hover_data=["Date"]
        )
        fig.show()

    .. plotly::
        :fig-vars: fig

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        # Get GPS Data
        gps = pd.read_csv('https://info.endaq.com/hubfs/data/mide-map-gps-data.csv')

        # Generate & Show Map
        fig = endaq.plot.gen_map(
            gps,
            lat="Latitude",
            lon="Longitude",
            color_by_column="Ground Speed",
            hover_data=["Date"]
        )
        fig.show()
    """
    if filter_positive_color_vals and color_by_column is not None:
        df_map = df_map[df_map[color_by_column] > 0]
    
    zoom = determine_plotly_map_zoom(lats=df_map[lat], lons=df_map[lon])
    center = get_center_of_coordinates(lats=df_map[lat], lons=df_map[lon])
    
    px.set_mapbox_access_token(mapbox_access_token)
    
    fig = px.scatter_mapbox(
        df_map,
        lat=lat,
        lon=lon,
        color=color_by_column,
        hover_data=hover_data,
        size_max=size_max,
        zoom=zoom + zoom_offset,
        center=center,
    )

    if mapbox_access_token is None:
        fig.update_layout(mapbox_style="stamen-terrain")

    return fig.update_layout(margin={"r": 20, "t": 20, "l": 20, "b": 0})
    

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

    num_slices = int(len(df) * utils.sample_spacing(df) / window)

    df_psd = psd.rolling_psd(df, num_slices=num_slices, octave_bins=bins_per_octave,
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
    psd_df = psd.welch(df, 1, scaling='spectrum')

    octave_psd_df = psd.to_octave(
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


def rolling_min_max_envelope(df: pd.DataFrame, desired_num_points: int = 2000, plot_as_bars: bool = True,
                             plot_title: str = "", opacity: float = 0.7,
                             colors_to_use: Optional[Container] = None) -> go.Figure:
    """
    A function to create a Plotly Figure to plot the data for each of the available data sub-channels, designed to
    reduce the number of points/data being plotted without minimizing the insight available from the plots.  It will
    plot either an envelope for rolling windows of the data (plotting the max and the min as line plots), or a bar based
    plot where the top of the bar (rectangle) is the highest value in the time window that bar spans, and the bottom of
    the bar is the lowest point in that time window (choosing between them is done with the `plot_as_bars` parameter).

    :param df: The dataframe of sub-channel data indexed by time stamps
    :param desired_num_points: The desired number of points to be plotted for each subchannel.  The number of points
     will be reduced from its original sampling rate by applying metrics (e.g. min, max) over sliding windows
     and then using that information to represent/visualize the data contained within the original data.  If less than
     the desired number of points are present, then a sliding window will NOT be used, and instead the points will be
     plotted as they were originally recorded  (also the subchannel will NOT be plotted as a bar based plot even if
     `plot_as_bars` was set to true).
    :param plot_as_bars: A boolean value indicating if the data should be visualized as a set of rectangles, where a
     shaded rectangle is used to represent the maximum and minimum values of the data during the time window
     covered by the rectangle.  These maximum and minimum values are visualized by the locations of the top and bottom
     edges of the rectangle respectively, unless the height of the rectangle would be 0, in which case a line segment
     will be displayed in its place.  If this parameter is `False`, two lines will be plotted for each
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


    Here's an example plotting a dataset with over 6 million points per channel

    .. code:: python3

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.graph_objects as go

        # Get Accel, 6M datapoints per axis
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',
            measurement_type='accel',
            time_mode='datetime')

        # Apply Highpass Filter
        accel = endaq.calc.filters.butterworth(accel, low_cutoff=2)

        # Generate Shaded Bar Plot of All Data
        fig = endaq.plot.rolling_min_max_envelope(accel)
        fig.show()

    .. plotly::
        :fig-vars: fig

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.graph_objects as go

        # Get Accel, 6M datapoints per axis
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',
            measurement_type='accel',
            time_mode='datetime')

        # Apply Highpass Filter
        accel = endaq.calc.filters.butterworth(accel, low_cutoff=2)

        # Generate Shaded Bar Plot of All Data
        fig = endaq.plot.rolling_min_max_envelope(accel)
        fig.show()

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
        round_freq: int = 2,
        waterfall_line_sequence: bool = True,
        zsmooth: str = "best",
        waterfall_line_color: str = '#EE7F27',
        min_median_max_line_color: str = '#6914F0',
) -> go.Figure:
    """
    Generate a 3D Plotly figure from a stacked spectrum to visualize how the frequency content changes over time
    
    :param df: the input dataframe with columns defining the frequency content, timestamps, values, and variables,
        see the following functions which provides outputs that would then be passed into this function as an input:

        *  :py:func:`~endaq.calc.fft.rolling_fft()`
        *  :py:func:`~endaq.calc.psd.rolling_psd()`
        *  :py:func:`~endaq.calc.shock.rolling_shock_spectrum()`
        *  :py:func:`~endaq.batch.GetDataBuilder.add_psd()`
        *  :py:func:`~endaq.batch.GetDataBuilder.add_pvss()`
    :param plot_type: the type of plot to display the spectrum, options are:

        *  `Heatmap`:  a 2D visualization with the color defining the z value (the default)
        *  `Surface`: similar to `Heatmap` but the z value is also projected "off the page"
        *  `Waterfall`: distinct lines are plotted per time slice in a 3D view
        *  `Animation`: a 2D display of the waterfall but the spectrum changes with animation frames
        *  `Peak`: per timestamp the peak frequency is determined and plotted against time
        *  `Lines`: the value in each frequency bin is plotted against time
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
    :param round_freq: number of decimals to round the frequency bins to, default is 3
    :param waterfall_line_sequence: if `True` the waterfall line colors are defined with a color scale,
        if `False` all lines will have the same color, default is `True`
    :param zsmooth: the Plotly smooth algorithm to use in the heatmap, default is `"best"` which looks ideal but
        `"fast"` will be more responsive, or `False` will attempt no smoothing
    :param waterfall_line_color: the color to use for all lines in the Waterfall plot if `waterfall_line_sequence` is
        `False`
    :param min_median_max_line_color: the color to use for the min, max, and median lines in the Animation, if set to
        `None` these lines won't be added, default is `'#6914F0'`
    :return: a Plotly figure visualizing the spectrum over time


    Here's a few examples from a dataset recorded with an enDAQ sensor on a motorcycle
    as it revved the engine which resulted in changing frequency content
    
    .. code:: python

        import pandas as pd
        import endaq

        # Set Theme
        endaq.plot.utilities.set_theme()

        # Get Vibration Data
        df_vibe = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-vibration-moving-frequency.csv', index_col=0)

        # Calculate a Rolling FFT
        fft = endaq.calc.fft.rolling_fft(df_vibe, num_slices=200, add_resultant=True)

        # Visualize the Rolling FFT as a Heatmap
        heatmap = endaq.plot.plots.spectrum_over_time(fft, plot_type='Heatmap', freq_max=200, var_to_process='Resultant')
        heatmap.show()

        # Plot the Peak Frequency vs Time
        peak = endaq.plot.plots.spectrum_over_time(fft, plot_type='Peak', freq_max=200, var_to_process='Resultant')
        peak.show()

        # Visualize as a Surface Plot
        surface = endaq.plot.plots.spectrum_over_time(fft, plot_type='Surface', freq_max=200, var_to_process='Resultant')
        surface.show()

        # Visualize as a Waterfall
        waterfall = endaq.plot.plots.spectrum_over_time(fft, plot_type='Waterfall', freq_max=200, var_to_process='Resultant')
        waterfall.show()

    .. plotly::
        :fig-vars: heatmap, peak, surface, waterfall

        import pandas as pd
        import endaq

        # Set Theme
        endaq.plot.utilities.set_theme()

        # Get Vibration Data
        df_vibe = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-vibration-moving-frequency.csv', index_col=0)

        # Calculate a Rolling FFT
        fft = endaq.calc.fft.rolling_fft(df_vibe, num_slices=200, add_resultant=True)

        # Visualize the Rolling FFT as a Heatmap
        heatmap = endaq.plot.plots.spectrum_over_time(fft, plot_type='Heatmap', freq_max=200, var_to_process='Resultant')
        heatmap.show()

        # Plot the Peak Frequency vs Time
        peak = endaq.plot.plots.spectrum_over_time(fft, plot_type='Peak', freq_max=200, var_to_process='Resultant')
        peak.show()

        # Visualize as a Surface Plot
        surface = endaq.plot.plots.spectrum_over_time(fft, plot_type='Surface', freq_max=200, var_to_process='Resultant')
        surface.show()

        # Visualize as a Waterfall
        waterfall = endaq.plot.plots.spectrum_over_time(fft, plot_type='Waterfall', freq_max=200, var_to_process='Resultant')
        waterfall.show()

    Here's another few examples with a longer dataset with DatetimeIndex of a car engine during a morning commute

    .. code:: python

        import pandas as pd
        import endaq

        # Set Theme
        endaq.plot.utilities.set_theme()

        # Get a Longer Dataset with DatetimeIndex
        engine = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/data/Commute.ide', measurement_type='accel',
                                                   time_mode='datetime')

        # Compute PSD
        psd = endaq.calc.psd.rolling_psd(engine, num_slices=500, add_resultant=True, octave_bins=12, fstart=4)

        # Visualize as a Heatmap
        heatmap2 = endaq.plot.plots.spectrum_over_time(psd, plot_type='Heatmap', var_to_process='Resultant', zsmooth='best',
                                                       log_freq=True, log_val=True)
        heatmap2.show()

        # Visualize as an Animation
        animation = endaq.plot.plots.spectrum_over_time(psd, plot_type='Animation', var_to_process='Resultant',
                                                        log_freq=True, log_val=True)
        animation.show()

        # Use Rolling PSD to Calculate RMS in Certain Frequency Bins
        rms = endaq.calc.psd.rolling_psd(engine, num_slices=500, scaling='rms', add_resultant=True,
                                         freq_splits=[1, 20, 60, 300, 3000])

        # Plot the RMS per Frequency Bin Over Time
        lines = endaq.plot.plots.spectrum_over_time(rms, plot_type='Lines', log_val=False, var_to_process='Resultant')
        lines.show()

        # Compute Pseudo Velocity at Specific Times
        pvss = endaq.calc.shock.rolling_shock_spectrum(engine, slice_width=2.0, add_resultant=True,
                                                       mode='pvss', init_freq=4, damp=0.05,
                                                       index_values=pd.DatetimeIndex(['2016-08-02 12:07:15',
                                                                                      '2016-08-02 12:08:01',
                                                                                      '2016-08-02 12:10:06'], tz='UTC'))

        # Visualize as a Waterfall
        waterfall2 = endaq.plot.plots.spectrum_over_time(pvss, plot_type='Waterfall', log_freq=True, log_val=True,
                                                         var_to_process='Resultant', waterfall_line_sequence=False)
        waterfall2.show()

    .. plotly::
        :fig-vars: heatmap2, animation, lines, waterfall2

        import pandas as pd
        import endaq

        # Set Theme
        endaq.plot.utilities.set_theme()

        # Get a Longer Dataset with DatetimeIndex
        engine = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/data/Commute.ide', measurement_type='accel',
                                                   time_mode='datetime')

        # Compute PSD
        psd = endaq.calc.psd.rolling_psd(engine, num_slices=500, add_resultant=True, octave_bins=12, fstart=4)

        # Visualize as a Heatmap
        heatmap2 = endaq.plot.plots.spectrum_over_time(psd, plot_type='Heatmap', var_to_process='Resultant', zsmooth='best',
                                                       log_freq=True, log_val=True)
        heatmap2.show()

        # Visualize as an Animation
        animation = endaq.plot.plots.spectrum_over_time(psd, plot_type='Animation', var_to_process='Resultant',
                                                        log_freq=True, log_val=True)
        animation.show()

        # Use Rolling PSD to Calculate RMS in Certain Frequency Bins
        rms = endaq.calc.psd.rolling_psd(engine, num_slices=500, scaling='rms', add_resultant=True,
                                         freq_splits=[1, 20, 60, 300, 3000])

        # Plot the RMS per Frequency Bin Over Time
        lines = endaq.plot.plots.spectrum_over_time(rms, plot_type='Lines', log_val=False, var_to_process='Resultant')
        lines.show()

        # Compute Pseudo Velocity at Specific Times
        pvss = endaq.calc.shock.rolling_shock_spectrum(engine, slice_width=2.0, add_resultant=True,
                                                       mode='pvss', init_freq=4, damp=0.05,
                                                       index_values=pd.DatetimeIndex(['2016-08-02 12:07:15',
                                                                                      '2016-08-02 12:08:01',
                                                                                      '2016-08-02 12:10:06'], tz='UTC'))

        # Visualize as a Waterfall
        waterfall2 = endaq.plot.plots.spectrum_over_time(pvss, plot_type='Waterfall', log_freq=True, log_val=True,
                                                         var_to_process='Resultant', waterfall_line_sequence=False)
        waterfall2.show()

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
        df = df.loc[df[freq_column] <= freq_max]
    if freq_min is not None:
        df = df.loc[df[freq_column] >= freq_min]

    # Remove 0s
    df = df.loc[df[val_column] > 0]

    # Check Length of Dataframe
    if len(df) > 100000:
        warnings.warn(
            "plot data is very large, may be unresponsive, suggest limiting frequency range and/or using less slices",
            RuntimeWarning,
        )

    # Create pivot table
    df_pivot = df.copy()
    first_step = df_pivot[freq_column].iloc[1] - df_pivot[freq_column].iloc[0]
    second_step = df_pivot[freq_column].iloc[2] - df_pivot[freq_column].iloc[1]
    if np.isclose(first_step, second_step):
        min_freq = np.round(df_pivot[freq_column].min(), round_freq)
        df_pivot[freq_column] = np.round(df_pivot[freq_column].to_numpy() / min_freq, 0) * min_freq
    df_pivot = df_pivot.pivot_table(columns=time_column, index=freq_column, values=val_column)

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
            fig = go.Figure(data=go.Heatmap(data_dict, zsmooth=zsmooth)).update_layout(
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
            fig.add_trace(go.Scatter(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
            ))

            # Add Min
            df_t = df_pivot.min(axis=1).dropna()
            fig.add_trace(go.Scatter(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
            ))

            # Add Median
            df_t = df_pivot.median(axis=1).dropna()
            fig.add_trace(go.Scatter(
                x=df_t.index,
                y=df_t,
                mode='lines',
                line_color=min_median_max_line_color,
                line_dash='dash',
            ))

    # Peak Frequencies
    elif plot_type == 'Peak':
        fig = px.scatter(
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


def _log_ticks(start, stop, spacing):
    """Create an array of values that would be the equivalent of tick marks in a log scaled plot"""
    start = np.floor(np.log10(start))
    stop = np.ceil(np.log10(stop))
    if spacing == 'coarse':
        ones = np.array([1])
    elif spacing == 'medium':
        ones = np.array([1, 2, 5])
    else:
        ones = np.linspace(1, 9, 9)
    output = ones * (10 ** start)
    for i in np.arange(start + 1, stop, 1):
        output = np.concatenate((output,
                                 ones * (10 ** i)))
    return np.concatenate((output, np.array([10 ** stop])))


def _add_df(fig, df, line_color, units):
    """Add lines for each column in a dataframe to an existing figure"""
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col].to_numpy(),
            mode="lines",
            line=dict(color=line_color, width=1, dash='solid'),
            showlegend=False,
            hoverinfo='skip',
            name=str(col) + units
        ))
    return fig


def pvss_on_4cp(
        df: pd.DataFrame,
        mode: typing.Literal["srs", "pvss"] = "srs",
        accel_units: str = "gravity",
        disp_units: str = "in",
        tick_spacing: typing.Literal["fine", "medium", "coarse"] = "medium",
        include_text: bool = True,
        size: typing.Optional[int] = None,
) -> go.Figure:
    """
    Given a shock response as a SRS or PVSS (see :py:func:`~endaq.calc.shock.shock_spectrum()`) return a plot of the
    pseudo velocity on a four coordinate plot (4CP or tripartite) that includes diagonal lines and hover labels to also
    signify the displacement and acceleration levels as a function of natural frequency.

    :param df: the input dataframe of shock response data, each column is plotted separately
    :param mode: the type of spectrum of the input dataframe, options are:

        *  `srs`:  default, shock response spectrum (SRS) which assumes has units of `accel_units`
        *  `pvss`: pseudo-velocity shock spectrum (PVSS) which assumes has units of `accel_units * s`
    :param accel_units: the units to display acceleration as, default is `"gravity"` which will be shortened to 'g'
        in labels, the unit conversion is handled using :py:func:`~endaq.calc.utils.convert_units()`
    :param disp_units: the units to display displacement as and velocity (divided by seconds), default is `"in"`,
        the unit conversion is handled using :py:func:`~endaq.calc.utils.convert_units()`
    :param tick_spacing: the spacing of each tick and corresponding diagonal line:

        *  `fine`:  order of magnitude, and linearly spaced ticks between each order of magnitude
        *  `medium`: default, order of magnitude, and then 2x and 5x between each order of magnitude, typical for Plotly,
        *  `coarse`: only the order of magnitude
    :param include_text: if `True` (default) add text labels to the diagonal lines
    :param size: the number of pixels to set the width and height of the figure to force it to be square,
        default is `None`
    :return: a Plotly figure of the PVSS on 4CP paper with hover information for acceleration and displacement


    Here's a few examples from a dataset recorded with an enDAQ sensor on a motorcycle
    as it revved the engine which resulted in changing frequency content

    .. code:: python3

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        # Get crash data
        df_crash = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-crash.csv',index_col=0)

        # Calculate SRS
        srs = endaq.calc.shock.shock_spectrum(df_crash, mode='srs', damp=0.05)

        # Generate 4CP Plot
        imp = endaq.plot.plots.pvss_on_4cp(srs, disp_units='in')
        imp.show()

        # Change Plot Theme
        endaq.plot.utilities.set_theme('endaq_light')

        # Generate 4CP Plot with Different Units
        met = endaq.plot.plots.pvss_on_4cp(srs, disp_units='mm', tick_spacing='coarse', size=500)
        met.show()

    .. plotly::
        :fig-vars: imp, met

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd

        # Get crash data
        df_crash = pd.read_csv('https://info.endaq.com/hubfs/data/motorcycle-crash.csv',index_col=0)

        # Calculate SRS
        srs = endaq.calc.shock.shock_spectrum(df_crash, mode='srs', damp=0.05)

        # Generate 4CP Plot
        imp = endaq.plot.plots.pvss_on_4cp(srs, disp_units='in')
        imp.show()

        # Change Plot Theme
        endaq.plot.utilities.set_theme('endaq_light')

        # Generate 4CP Plot with Different Units
        met = endaq.plot.plots.pvss_on_4cp(srs, disp_units='mm', tick_spacing='coarse', size=500)
        met.show()

    """
    # Get and apply unit conversion
    accel_2_disp = utils.convert_units(units_in=accel_units, units_out=disp_units + '/s^2')
    df = df.copy() * accel_2_disp

    # Generate pseudo velocity data if srs is provided
    if mode == 'srs':
        df = df.div(2 * np.pi * df.index.to_series(), axis=0)

    # Generate acceleration & displacement dataframes
    accel = df.mul(2 * np.pi * df.index.to_series(), axis=0) / accel_2_disp
    disp = df.div(2 * np.pi * df.index.to_series(), axis=0)

    # Specify accel label
    accel_label = accel_units
    if accel_label == 'gravity':
        accel_label = 'g'

    # Get diagonal line information
    pvss_ticks = _log_ticks(
        start=10 ** np.floor(np.log10(df.min(axis=1).min())),
        stop=10 ** np.ceil(np.log10(df.max(axis=1).max())),
        spacing=tick_spacing
    )
    accel_ticks = _log_ticks(
        start=pvss_ticks[0] * df.index[0] * 2 * np.pi / accel_2_disp,
        stop=pvss_ticks[-1] * df.index[-1] * 2 * np.pi / accel_2_disp,
        spacing=tick_spacing
    )
    disp_ticks = _log_ticks(
        start=pvss_ticks[0] / (df.index[-1] * 2 * np.pi),
        stop=pvss_ticks[-1] / (df.index[0] * 2 * np.pi),
        spacing=tick_spacing
    )
    freqs = np.append(pvss_ticks[-1] / disp_ticks[0] / (2 * np.pi), df.index)
    freqs = np.append(freqs, pvss_ticks[-1] / disp_ticks[-1] / (2 * np.pi))
    accel_tick_df = pd.DataFrame(
        np.outer(1 / (freqs * 2 * np.pi), accel_ticks * accel_2_disp),
        index=freqs,
        columns=accel_ticks
    )
    disp_tick_df = pd.DataFrame(
        np.outer((freqs * 2 * np.pi), disp_ticks),
        index=freqs,
        columns=disp_ticks
    )

    # Generate figure & add shock spectrum data to plots
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scattergl(
            x=df.index.to_numpy(),
            y=df[c].to_numpy(),
            name=c,
            customdata=pd.DataFrame({
                    'Freq': df.index.to_numpy(),
                    'Accel': accel[c].to_numpy(),
                    'PV': df[c].to_numpy(),
                    'Disp': disp[c].to_numpy()}
            ),
            hovertemplate=c + '<br><br>' +
                          'Natural Frequency (Hz): %{customdata[0]:.2f}<br><br>' +
                          'Acceleration (' + accel_label + '): %{customdata[1]:.4E}<br>' +
                          'Pseudo-Velocity (' + disp_units + '/s): %{customdata[2]:.4E}<br>' +
                          'Displacement (' + disp_units + '): %{customdata[3]:.4E}<br><extra></extra>'
        ))

    # Add diagonal lines
    fig = _add_df(fig, accel_tick_df, fig.layout.template.layout.xaxis.gridcolor, accel_label)
    fig = _add_df(fig, disp_tick_df, fig.layout.template.layout.xaxis.gridcolor, disp_units)

    # Add text
    if include_text:
        dt = disp_tick_df.columns
        if tick_spacing == 'fine':
            dt = dt[np.remainder(np.log10(dt), 1) == 0]
        for disp in dt:
            disp_str = str(disp) + " " + disp_units
            if disp >= 1:
                disp_str = f"{disp:,.0f}" + " " + disp_units
            fig.add_annotation(
                x=np.log10(pvss_ticks[-1] / (disp * 2 * np.pi)),
                y=1,
                text=disp_str,
                yref='paper',
                xanchor='left',
                yanchor='bottom',
                textangle=-45,
                showarrow=False)
        at = accel_tick_df.columns
        if tick_spacing == 'fine':
            at = at[np.remainder(np.log10(at), 1) == 0]
        for accel in at:
            accel_str = str(accel) + " " + accel_label
            if accel >= 1:
                accel_str = f"{accel:,.0f}" + " " + accel_label
            fig.add_annotation(
                x=1,
                y=np.log10(accel / (df.index[-1] * 2 * np.pi) * accel_2_disp),
                text=accel_str,
                xref='paper',
                xanchor='left',
                yanchor='top',
                textangle=45,
                showarrow=False)

    return fig.update_layout(
        yaxis_type='log',
        xaxis_type='log',
        yaxis_range=np.log10([pvss_ticks[0], pvss_ticks[-1]]),
        xaxis_range=np.log10([df.index[0], df.index[-1]]),
        yaxis_title_text='Pseudo Velocity (' + disp_units + '/s)',
        xaxis_title_text='Natural Frequency (Hz)',
        width=size,
        height=size
    )


def table_plot(
        table: pd.DataFrame,
        num_round: typing.Optional[int] = 2,
        row_size: typing.Optional[int] = None,
        font_color: typing.Optional[str] = None,
        bg_color: typing.Optional[str] = None,
        line_color: typing.Optional[str] = None
) -> go.Figure:
    """
    Generate a Plotly figure from a Pandas dataframe that is styled consistent with the current Plotly template

    :param table: the input dataframe to generate the plot from
    :param num_round: the precision to round all numbers (if any) in `table` before displaying
    :param row_size: the size for each row, if `None` it will set this to 2 times the default Plotly template font size
    :param font_color: the color of all cell's text and the background color for the header, if `None` it will set this
        to the default Plotly font color
    :param bg_color: the color of all cell's background color and the text for the header, if `None` it will set this
        to the default Plotly background color
    :param line_color: the color for the lines in the table, if `None` it will set this
        to the default Plotly grid color
    :return: a Plotly table figure with all content from the dataframe


    Here's an example to generate a table from the metrics calculated with :py:func:`~endaq.calc.stats.shock_vibe_metrics()`

    .. code:: python3

        import endaq
        endaq.plot.utilities.set_theme('endaq_light')
        import pandas as pd

        # Get Acceleration Data
        accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

        # Calculate Metrics
        metrics = endaq.calc.stats.shock_vibe_metrics(accel, include_resultant=False, freq_splits=[0, 65, 300, None])

        # Generate Plot to Show Metrics as a Table
        light_table = endaq.plot.table_plot(metrics)
        light_table.show()

        # Change Theme
        endaq.plot.utilities.set_theme('endaq')

        # Regenerate Plot to Show Metrics as a Table
        dark_table = endaq.plot.table_plot(metrics, num_round=4)
        dark_table.show()

    .. plotly::
        :fig-vars: light_table, dark_table

        import endaq
        endaq.plot.utilities.set_theme('endaq_light')
        import pandas as pd

        # Get Acceleration Data
        accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

        # Calculate Metrics
        metrics = endaq.calc.stats.shock_vibe_metrics(accel, include_resultant=False, freq_splits=[0, 65, 300, None])

        # Generate Plot to Show Metrics as a Table
        light_table = endaq.plot.table_plot(metrics)
        light_table.show()

        # Change Theme
        endaq.plot.utilities.set_theme('endaq')

        # Regenerate Plot to Show Metrics as a Table
        dark_table = endaq.plot.table_plot(metrics, num_round=4)
        dark_table.show()

    """
    # Generate a Plotly Figure, then determine colors and sizes
    fig_template = go.Figure()
    if row_size is None:
        row_size = fig_template.layout.template.layout.font.size
        # Some Plotly templates don't specify this so we need to check if None again
        if row_size is None:
            row_size = 30
        else:
            row_size *= 2
    if bg_color is None:
        bg_color = fig_template.layout.template.layout.plot_bgcolor
    if font_color is None:
        font_color = fig_template.layout.template.layout.font.color
    if line_color is None:
        line_color = fig_template.layout.template.layout.xaxis.gridcolor

    # Generate Plot
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=list(table.columns),
                    fill_color=font_color,
                    font_color=bg_color,
                    height=row_size),
        cells=dict(values=table.round(num_round).transpose().values.tolist(),
                   fill_color=bg_color,
                   height=row_size,
                   line_color=line_color),
    )
    ])
    return fig_table


def table_plot_from_ide(
        doc: idelib.dataset.Dataset = None,
        name: str = None,
        **kwargs
) -> go.Figure:
    """
    Generate a Plotly figure from a .IDE file using :py:func:`~endaq.plot.table_plot()` and
    :py:func:`~endaq.ide.get_channel_table()` while also displaying the device serial number, part number, and the
    date of the recording

    :param doc: A `idelib.dataset.Dataset`
    :param name: The plot title to add, if `None` then no title is added
    :param kwargs: Other parameters to pass directly to :py:func:`~endaq.plot.table_plot()`
    :return: a Plotly table figure with all content from the dataframe


    Here's an example to generate a table from the metrics calculated with :py:func:`~endaq.calc.stats.shock_vibe_metrics()`

    .. code:: python3

        import endaq
        endaq.plot.utilities.set_theme()

        doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/All-Channels.ide')
        fig = endaq.plot.table_plot_from_ide(doc=doc, name='Example File')
        fig.show()

    .. plotly::
        :fig-vars: fig

        import endaq
        endaq.plot.utilities.set_theme()

        doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/All-Channels.ide')
        fig = endaq.plot.table_plot_from_ide(doc=doc, name='Example File')
        fig.show()

    """
    # Get File Table
    file_table = get_channel_table(doc).data
    table = file_table[['name', 'type', 'units', 'samples', 'rate']].copy()

    # Get Figure
    fig_table = table_plot(table=table, **kwargs)

    # Add Title
    recorder_name = doc.recorderInfo.get('RecorderName', '')
    serial = doc.recorderInfo.get('RecorderSerial', '')
    part_no = doc.recorderInfo.get('PartNumber', '')
    date = pd.to_datetime(doc.lastUtcTime, unit='s')
    table_title = f'{recorder_name} (# {serial}, a {part_no})<br>Recorded on {date}'
    text_size = fig_table.layout.template.layout.font.size
    if text_size is None:
        text_size = 30
    else:
        text_size *= 2
    t_margin = text_size * 2 + 20
    if name is not None:
        table_title = name + '<br>' + table_title
        t_margin += text_size

    return fig_table.update_layout(
        title_text=table_title,
        title_y=.95, title_yanchor='top',
        margin=dict(l=20, r=20, t=t_margin, b=0))
