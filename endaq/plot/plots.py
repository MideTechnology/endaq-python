from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from typing import Optional
from collections.abc import Container

from endaq.calc import sample_spacing
from endaq.calc.psd import to_octave, welch

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
    :param color_col: The column name in the given dataframe (as merged_df) that is used to color
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
    :param color_col: The column name in the given dataframe (as merged_df) that is used to color
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
                       ) -> go.Figure:
    """
    Produces an octave spectrogram of the given data.

    :param df: The dataframe of sensor data.  This must only have 1 column.
    :param window: The time window for each of the columns in the spectrogram
    :param bins_per_octave: The number of frequency bins per octave
    :param freq_start: The center of the first frequency bin
    :param max_freq: The maximum frequency to plot
    :param db_scale: If the spectrogram should be log-scaled for visibility (with 10*log10(x))
    :param log_scale_y_axis: If the y-axis of the plot should be log scaled
    :return: a tuple containing:

        - the frequency bins
        - the time bins
        - the spectrogram data
        - the corresponding plotly figure
    """
    if len(df.columns) != 1:
        raise ValueError("The parameter 'df' must have only one column of data!")

    ary = df.values.squeeze()

    fs = 1/sample_spacing(df)#(len(df) - 1) / (df.index[-1] - df.index[0])
    N = int(fs * window) #Number of points in the fft
    w = signal.blackman(N, False)
    
    freqs, bins, Pxx = signal.spectrogram(ary, fs, window=w, nperseg=N, noverlap=0)

    time_dfs = [pd.DataFrame({bins[j]: Pxx[:, j]}, index=freqs) for j in range(len(bins))]

    octave_dfs = list(map(lambda x: to_octave(x, freq_start, octave_bins=bins_per_octave, agg='sum'), time_dfs))
    
    combined_df = pd.concat(octave_dfs, axis=1)
    
    freqs = combined_df.index.values
    Pxx = combined_df.values
    
    include_freqs_mask = freqs <= max_freq
    Pxx = Pxx[include_freqs_mask]
    freqs = freqs[include_freqs_mask]

    if db_scale:
        Pxx = 10 * np.log10(Pxx)

    trace = [go.Heatmap(x=bins, y=freqs, z=Pxx, colorscale='Jet')]
    layout = go.Layout(
        yaxis={'title': 'Frequency (Hz)'},
        xaxis={'title': 'Time (s)'},
    )

    fig = go.Figure(data=trace, layout=layout)
    
    if log_scale_y_axis:
        fig.update_yaxes(type="log")

    fig.update_traces(showscale=False)

    data_df = pd.DataFrame(Pxx, index=freqs, columns=bins)

    return data_df, fig
    

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
