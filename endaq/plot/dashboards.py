from __future__ import annotations

from collections.abc import Container
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Optional


def rolling_enveloped_dashboard(
    channel_df_dict: dict, desired_num_points: int = 250, num_rows: Optional[int] = None,
    num_cols: Optional[int] = 3, width_for_subplot_row: int = 400, height_for_subplot_row: int = 400,
    subplot_colors: Optional[Container] = None, min_points_to_plot: int = 1, plot_as_bars: bool = False,
    plot_full_single_channel: bool = False, opacity: float = 1.0, y_axis_bar_plot_padding: float = 0.06
) -> go.Figure:
    """
    A function to create a Plotly Figure with sub-plots for each of the available data sub-channels, designed to reduce
    the number of points/data being plotted without minimizing the insight available from the plots.  It will plot
    either an envelope for rolling windows of the data (plotting the max and the min as line plots), or a bar based
    plot where the top of the bar (rectangle) is the highest value in the time window that bar spans, and the bottom of
    the bar is the lowest point in that time window (choosing between them is done with the `plot_as_bars` parameter).


    :param channel_df_dict: A dictionary mapping channel names to Pandas DataFrames of that channels data
    :param desired_num_points: The desired number of points to be plotted in each subplot.  The number of points
     will be reduced from its original sampling rate by applying metrics (e.g. min, max) over sliding windows
     and then using that information to represent/visualize the data contained within the original data.  If less than
     the desired number of points are present, then a sliding window will NOT be used, and instead the points will be
     plotted as they were originally recorded  (also the subplot will NOT be plotted as a bar based plot even if
     `plot_as_bars` was set to true).
    :param num_rows: The number of columns of subplots to be created inside the Plotly figure. If `None` is given, (then
     `num_cols` must not be `None`), then this number will automatically be determined by what's needed.  If more rows
     are specified than are needed, the number of rows will be reduced to the minimum needed to contain all the subplots
    :param num_cols: The number of columns of subplots to be created inside the Plotly figure.  See the description of
     the `num_rows` parameter for more details on this parameter, and how the two interact.  This also follows the same
     approach to handling `None` when given
    :param width_for_subplot_row: The width of the area used for a single subplot (in pixels).
    :param height_for_subplot_row: The height of the area used for a single subplot (in pixels).
    :param subplot_colors: An "array-like" object of strings containing colors to be cycled through for the subplots.
     If `None` is given (which is the default), then the `colorway` variable in Plotly's current theme/template will
     be used to color the data on each of the subplots uniquely, repeating from the start of the `colorway` if
     all colors have been used.
    :param min_points_to_plot: The minimum number of data points required to be present to create a subplot for a
     channel/subchannel (NOT including `NaN` values).
    :param plot_as_bars: A boolean value indicating if the plot should be visualized as a set of rectangles, where a
     shaded rectangle is used to represent the maximum and minimum values of the data during the time window
     covered by the rectangle.  These maximum and minimum values are visualized by the locations of the top and bottom
     edges of the rectangle respectively, unless the height of the rectangle would be 0, in which case a line segment
     will be displayed in its place.  If this parameter is `False`, two lines will be plotted for each
     of the subplots in the figure being created, creating an "envelope" around the data.  An "envelope" around the
     data consists of a line plotted for the maximum values contained in each of the time windows, and another line
     plotted for the minimum values.  Together these lines create a boundary which contains all the data points
     recorded in the originally recorded data.
    :param plot_full_single_channel: If instead of a dashboard of subplots a single plot with multiple sub-channels
     should be created.  If this is True, only one (key, value) pair can be given for the `channel_df_dict` parameter
    :param opacity: The opacity to use for plotting bars/lines
    :param y_axis_bar_plot_padding: Due to some unknown reason the bar subplots aren't having their y axis ranges
     automatically scaled so this is the ratio of the total y-axis data range to pad both the top and bottom of the
     y axis with.  The default value is the one it appears Plotly uses as well.
    :return: The Plotly Figure containing the subplots of sensor data (the 'dashboard')
    """
    if not (num_rows is None or isinstance(num_rows, (int, np.integer))):
        raise TypeError(f"`num_rows` is of type `{type(num_rows)}`, which is not allowed.  " 
                        "`num_rows` can either be `None` or some type of integer.")
    elif not (num_cols is None or isinstance(num_cols, (int, np.integer))):
        raise TypeError(f"`num_cols` is of type `{type(num_cols)}`, which is not allowed.  "
                        "`num_cols` can either be `None` or some type of integer.")

    # I'm pretty sure the below is correct and it appears to work correctly, but it'd be nice if someone could
    # double check this logic

    # This removes any channels with less than `min_points_to_plot` data points per sub-channel, and removes any
    # sub-channels which have less than `min_points_to_plot` non-NaN data points
    channel_df_dict = {
        k: v.drop(columns=v.columns[v.notna().sum(axis=0) < min_points_to_plot])
            for (k, v) in channel_df_dict.items() if v.shape[0] >= min_points_to_plot}

    subplot_titles = [' '.join((k, col)) for (k, v) in channel_df_dict.items() for col in v.columns]

    if num_rows is None and num_cols is None:
        raise TypeError("Both `num_rows` and `num_columns` were given as `None`!  "
                        "A maximum of one of these two parameters may be given as None.")
    elif num_rows is None:
        num_rows = 1 + (len(subplot_titles) - 1) // num_cols
    elif num_cols is None:
        num_cols = 1 + (len(subplot_titles) - 1) // num_rows
    elif len(subplot_titles) > num_rows * num_cols:
        raise ValueError("The values given for `num_rows` and `num_columns` result in a maximum "
                        f"of {num_rows * num_cols} avaialable sub-plots, but {len(subplot_titles)} subplots need "
                        "to be plotted!   Try setting one of these variables to `None`, it will then "
                        "automatically be set to the optimal number of rows/columns.")
    else:
        num_rows = 1 + (len(subplot_titles) - 1) // num_cols
        num_cols = int(np.ceil(len(subplot_titles)/num_rows))

    if subplot_colors is None:
        colorway = pio.templates[pio.templates.default]['layout']['colorway']
    else:
        colorway = subplot_colors

    if plot_full_single_channel:
        if len(channel_df_dict) != 1:
            raise ValueError("The 'channel_df_dict' parameter must be length 1 when "
                             "'plot_full_single_channel' is set to true!")

        num_rows = 1
        num_cols = 1
        fig = go.Figure(layout_title_text=list(channel_df_dict.keys())[0])
    else:
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            figure=go.Figure(
                layout_height=height_for_subplot_row * num_rows,
                layout_width=width_for_subplot_row * num_cols,
            ),
        )

    # A counter to keep track of which subplot is currently being worked on
    subplot_num = 0

    # A dictionary to be used to modify the Plotly Figure layout all at once after all the elements have been added
    layout_changes_to_make = {}

    for channel_data in channel_df_dict.values():
        window = int(np.around((channel_data.shape[0]-1) / desired_num_points, decimals=0))

        # If a window size of 1 is determined, it sets the stride to 1 so we don't get an error as a result of the
        # 0 length stride
        stride = 1 if window == 1 else window - 1

        rolling_n = channel_data.rolling(window)
        min_max_tuple = (rolling_n.min()[::stride], rolling_n.max()[::stride])
        min_max_equal = min_max_tuple[0] == min_max_tuple[1]

        is_nan_min_max_mask = np.logical_not(
            np.logical_and(
                pd.isnull(min_max_tuple[0]),
                pd.isnull(min_max_tuple[1])))

        # If it's going to be plotted as bars, force its time stamps to be uniformly spaced
        # so that the bars don't have any discontinuities in the X-axis
        if len(channel_data) >= desired_num_points and plot_as_bars:
            if isinstance(channel_data.index, pd.core.indexes.datetimes.DatetimeIndex):
                new_index = pd.date_range(
                    channel_data.index.values[0],
                    channel_data.index.values[-1],
                    periods=len(channel_data),
                )
            else:
                new_index = np.linspace(
                    channel_data.index.values[0],
                    channel_data.index.values[-1],
                    num=len(channel_data),
                )

            channel_data.set_index(new_index)

        # Loop through each of the sub-channels, and their respective '0-height rectangle mask'
        for subchannel_name, cur_min_max_equal in min_max_equal[channel_data.columns].iteritems():

            traces = []
            cur_color = colorway[subplot_num % len(colorway)]
            cur_subchannel_non_nan_mask = is_nan_min_max_mask[subchannel_name]

            # If there are less data points than the desired number of points
            # to be plotted, just plot the data as a line plot
            if len(channel_data) < desired_num_points:
                not_nan_mask = np.logical_not(pd.isnull(channel_data[subchannel_name]))
                traces.append(
                    go.Scatter(
                        x=channel_data.index[not_nan_mask],
                        y=channel_data.loc[not_nan_mask, subchannel_name],
                        name=subchannel_name,
                        opacity=opacity,
                        line_color=cur_color,
                        showlegend=plot_full_single_channel,
                    )
                )
            # If it's going to plot the data with bars
            elif plot_as_bars:
                # If there are any 0-height rectangles
                if np.any(cur_min_max_equal):
                    equal_data_df = min_max_tuple[0].loc[cur_min_max_equal.values, subchannel_name]

                    # Half of the sampling period
                    half_dt = np.diff(cur_min_max_equal.index[[0, -1]])[0] / (2 * (len(cur_min_max_equal) - 1))

                    # Initialize the arrays we'll use for creating line segments where
                    # rectangles would have 0 width so it will end up formatted as follows
                    # (duplicate values for y since line segements are horizontal):
                    # x = [x1, x2, None, x3, x4, None, ...]
                    # y = [y12, y12, None, y34, y34, None, ...]
                    x_patch_line_segs = np.repeat(equal_data_df.index.values, 3)
                    y_patch_line_segs = np.repeat(equal_data_df.values, 3)

                    # All X axis values are the same, but these values are supposed to represent pairs of start and end
                    # times for line segments, so the time stamp is shifted half its duration backwards for the start
                    # time, and half its duration forward for the end time
                    x_patch_line_segs[::3] -= half_dt
                    x_patch_line_segs[1::3] += half_dt

                    # This is done every third value so that every two pairs of points is unconnected from eachother,
                    # since the (None, None) point will not connect to either the point before it nor behind it
                    x_patch_line_segs[2::3] = None
                    y_patch_line_segs[2::3] = None

                    traces.append(
                        go.Scatter(
                            x=x_patch_line_segs,
                            y=y_patch_line_segs,
                            name=subchannel_name,
                            opacity=opacity,
                            mode='lines',
                            line_color=cur_color,
                            showlegend=plot_full_single_channel,
                        )
                    )

                min_data_point = np.min(min_max_tuple[0][subchannel_name])
                max_data_point = np.max(min_max_tuple[1][subchannel_name])

                y_padding = (max_data_point - min_data_point) * y_axis_bar_plot_padding

                traces.append(
                    go.Bar(
                        x=min_max_tuple[0].index[cur_subchannel_non_nan_mask],
                        y=(min_max_tuple[1].loc[cur_subchannel_non_nan_mask, subchannel_name] -
                           min_max_tuple[0].loc[cur_subchannel_non_nan_mask, subchannel_name]),
                        marker_color=cur_color,
                        opacity=opacity,
                        marker_line_width=0,
                        base=min_max_tuple[0].loc[cur_subchannel_non_nan_mask, subchannel_name],
                        showlegend=plot_full_single_channel,
                        name=subchannel_name,
                    )
                )

                # Adds a (key, value) pair to the dict for setting this subplot's Y-axis display range (applied later)
                min_y_range = min_data_point - y_padding
                max_y_range = max_data_point + y_padding
                y_axis_id = f'yaxis{1 + subplot_num}_range'
                if plot_full_single_channel:
                    y_axis_id = 'yaxis_range'
                    if layout_changes_to_make:
                        min_y_range = min(min_y_range, layout_changes_to_make[y_axis_id][0])
                        max_y_range = max(max_y_range, layout_changes_to_make[y_axis_id][1])

                layout_changes_to_make[y_axis_id] = [min_y_range, max_y_range]
            else:
                for cur_df in min_max_tuple:
                    traces.append(
                        go.Scatter(
                            x=cur_df.index[cur_subchannel_non_nan_mask],
                            y=cur_df.loc[cur_subchannel_non_nan_mask, subchannel_name],
                            name=subchannel_name,
                            opacity=opacity,
                            line_color=cur_color,
                            showlegend=plot_full_single_channel,
                        )
                    )

            # Add the traces created for the current subchannel of data to the plotly figure
            if plot_full_single_channel:
                fig.add_traces(traces)
            else:
                fig.add_traces(
                    traces,
                    rows=1 + subplot_num // num_cols,
                    cols=1 + subplot_num % num_cols,
                )

            subplot_num += 1

    fig.update_layout(
        **layout_changes_to_make,
        bargap=0,
        barmode='overlay'
    )
    return fig


def rolling_metric_dashboard(channel_df_dict: dict, desired_num_points: int = 250, num_rows: Optional[int] = None,
             num_cols: Optional[int] = 3, rolling_metrics_to_plot: tuple = ('mean', 'std'),
             metric_colors: Optional[Container] = None, width_for_subplot_row: int = 400,
             height_for_subplot_row: int = 400) -> go.Figure:
    """
    A function to create a dashboard of subplots of the given data, plotting a set of rolling metrics.

    :param channel_df_dict: A dictionary mapping channel names to Pandas DataFrames of that channels data
    :param desired_num_points:  The desired number of points to be plotted in each subplot.  The number of points
     will be reduced from its original sampling rate by applying metrics (e.g. min, max) over sliding windows
     and then using that information to represent/visualize the data contained within the original data.  If less than
     the desired number of points are present, then a sliding window will NOT be used, and instead the points will be
     plotted as they were originally recorded  (also the subplot will NOT be plotted as a bar based plot even if
     `plot_as_bars` was set to true).
    :param num_rows: The number of columns of subplots to be created inside the Plotly figure. If `None` is given, (then
     `num_cols` must not be `None`), then this number will automatically be determined by what's needed.  If more rows
     are specified than are needed, the number of rows will be reduced to the minimum needed to contain all the subplots
    :param num_cols: The number of columns of subplots to be created inside the Plotly figure.  See the description of
     the `num_rows` parameter for more details on this parameter, and how the two interact.  This also follows the same
     approach to handling `None` when given
    :param rolling_metrics_to_plot: A tuple of strings which indicate what rolling metrics to plot for each subchannel.
     The options are ['mean', 'std', 'absolute max', 'rms'] which correspond to the mean, standard deviation, maximum of
     the absolute value, and root-mean-square.
    :param metric_colors: An "array-like" object of strings containing colors to be cycled through for the metrics.
     If `None` is given (which is the default), then the `colorway` variable in Plotly's current theme/template will
     be used to color the metric data, repeating from the start of the `colorway` if all colors have been used.
     The first value corresponds to the color if not enough points of data exist for a rolling metric,
     and the others correspond to the metric in `rolling_metrics_to_plot` in the same order they are given
    :param width_for_subplot_row: The width of the area used for a single subplot (in pixels).
    :param height_for_subplot_row: The height of the area used for a single subplot (in pixels).
    :return: The Plotly Figure containing the subplots of sensor data (the 'dashboard')
    """
    if len(rolling_metrics_to_plot) == 0:
        raise ValueError("At least one rolling metric must be specified in `rolling_metrics_to_plot`!")

    subplot_titles = [' '.join((k, col)) for (k, v) in channel_df_dict.items() for col in v.columns]

    if num_rows is None and num_cols is None:
        raise TypeError("Both `num_rows` and `num_columns` were given as `None`!  "
                        "A maximum of one of these two parameters may be given as None.")
    elif num_rows is None:
        num_rows = 1 + (len(subplot_titles) - 1) // num_cols
    elif num_cols is None:
        num_cols = 1 + (len(subplot_titles) - 1) // num_rows
    elif len(subplot_titles) > num_rows * num_cols:
        raise ValueError("The values given for `num_rows` and `num_columns` result in a maximum "
                         f"of {num_rows * num_cols} avaialable sub-plots, but {len(subplot_titles)} subplots need "
                         "to be plotted!   Try setting one of these variables to `None`, it will then "
                         "automatically be set to the optimal number of rows/columns.")
    else:
        num_rows = 1 + (len(subplot_titles) - 1) // num_cols
        num_cols = int(np.ceil(len(subplot_titles)/num_rows))

    if metric_colors is None:
        colorway = pio.templates[pio.templates.default]['layout']['colorway']
    else:
        colorway = metric_colors
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)
    subplot_num = 0
    for channel_name, channel_data in channel_df_dict.items():
        n = int(channel_data.shape[0] / desired_num_points)

        if n > 0:
            time = channel_data.rolling(n).mean().iloc[::n].index
        else:
            time = channel_data.index

        for c_i, (subchannel_name, subchannel_data) in enumerate(channel_data.iteritems()):
            for metric in rolling_metrics_to_plot:
                if n == 0:
                    data = subchannel_data.values
                    name = subchannel_name
                    color = colorway[0]
                elif metric == 'mean':
                    data = subchannel_data.rolling(n).mean().iloc[::n]
                    name = 'Mean'
                    color = colorway[1]
                elif metric == 'std':
                    data = subchannel_data.rolling(n).std().iloc[::n]
                    name = 'STD Dev'
                    color = colorway[2]
                elif metric == 'absolute max':
                    data = subchannel_data.abs().rolling(n).max().iloc[::n]
                    name = 'Max'
                    color = colorway[3]
                elif metric == 'rms':
                    data = subchannel_data.pow(2).rolling(n).mean().apply(np.sqrt, raw=True).iloc[::n]
                    name = 'RMS'
                    color = colorway[4]
                else:
                    raise ValueError(f"metric given to `rolling_metrics_to_plot` is not valid!  Was given {metric}"
                                     " which is not in the allowed options of ['mean', 'std', 'absolute max', 'rms']")
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=data,
                        name=name,
                        line=dict(color=color)),
                    row=1 + subplot_num // num_cols,
                    col=1 + subplot_num % num_cols,
                )

                if n == 0:
                    break

            subplot_num += 1

    return fig.update_layout(
        width=num_cols * width_for_subplot_row,
        height=num_rows * height_for_subplot_row,
        showlegend=False,
    )






if __name__ == '__main__':
    import endaq.ide
    from utilities import set_theme
    set_theme()

    file_urls = ['https://info.endaq.com/hubfs/data/surgical-instrument.ide',
                 'https://info.endaq.com/hubfs/data/97c3990f-Drive-Home_70-1616632444.ide',
                 'https://info.endaq.com/hubfs/data/High-Drop.ide',
                 'https://info.endaq.com/hubfs/data/HiTest-Shock.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home_01.ide',
                 'https://info.endaq.com/hubfs/data/Tower-of-Terror.ide',
                 'https://info.endaq.com/hubfs/data/Punching-Bag.ide',
                 'https://info.endaq.com/hubfs/data/Gun-Stock.ide',
                 'https://info.endaq.com/hubfs/data/Seat-Base_21.ide',
                 'https://info.endaq.com/hubfs/data/Seat-Top_09.ide',
                 'https://info.endaq.com/hubfs/data/Bolted.ide',
                 'https://info.endaq.com/hubfs/data/Motorcycle-Car-Crash.ide',
                 'https://info.endaq.com/hubfs/data/train-passing.ide',
                 'https://info.endaq.com/hubfs/data/baseball.ide',
                 'https://info.endaq.com/hubfs/data/Clean-Room-VC.ide',
                 'https://info.endaq.com/hubfs/data/enDAQ_Cropped.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home_07.ide',
                 'https://info.endaq.com/hubfs/data/ford_f150.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home.ide',
                 'https://info.endaq.com/hubfs/data/Mining-Data.ide',
                 'https://info.endaq.com/hubfs/data/Mide-Airport-Drive-Lexus-Hybrid-Dash-W8.ide']

    for j in [4]:
        doc = endaq.ide.get_doc(file_urls[j])
        table = endaq.ide.get_channel_table(doc)

        # (IMPORTANT NOTE) The use of this as a dictionary is dependent on it maintaining being 'insertion ordered',
        # which is a thing in Python 3.7 (may have existed in a different way in python 3.6, but I'm not sure)
        CHANNEL_DFS = {
            doc.channels[ch].name: endaq.ide.to_pandas(doc.channels[ch], time_mode='seconds') for ch in doc.channels}

        SINGLE_CHANNEL = r'40g DC Acceleration'
        JUST_ACCEL_DFS = {SINGLE_CHANNEL: CHANNEL_DFS[SINGLE_CHANNEL]}

        # Examples
        rolling_enveloped_dashboard(
            JUST_ACCEL_DFS,
            plot_full_single_channel=True,
        ).show()

        rolling_enveloped_dashboard(
            JUST_ACCEL_DFS,
            plot_as_bars=True,
            plot_full_single_channel=True,
        ).show()

        rolling_enveloped_dashboard(
            CHANNEL_DFS,
            desired_num_points=100,
            min_points_to_plot=10,
            plot_as_bars=True,
            height_for_subplot_row=600,
            width_for_subplot_row=600,
            num_cols=2,
            num_rows=None,
        ).show()

        rolling_enveloped_dashboard(
            CHANNEL_DFS,
            desired_num_points=1000,
            num_rows=2,
            num_cols=9999999,
            width_for_subplot_row=250,
            height_for_subplot_row=250,
        ).show()

        rolling_enveloped_dashboard(
            CHANNEL_DFS,
            desired_num_points=1000,
            num_rows=9999999,
            num_cols=2,
            width_for_subplot_row=250,
            height_for_subplot_row=250,
        ).show()

        rolling_enveloped_dashboard(
            CHANNEL_DFS,
            desired_num_points=1000,
            min_points_to_plot=10,
            plot_as_bars=True,
        ).show()

        rolling_enveloped_dashboard(
            CHANNEL_DFS,
            desired_num_points=1000,
            min_points_to_plot=10,
            num_rows=999999,
            num_cols=4,
        ).show()

        rolling_metric_dashboard(
            CHANNEL_DFS,
            rolling_metrics_to_plot=('mean', 'absolute max', 'std'),
        ).show()

        print(str(j) + " done!")
