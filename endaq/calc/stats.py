from __future__ import annotations

import typing  # for `SupportsIndex`, which is Python3.8+ only
from typing import Optional, Union
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go
import plotly.express as px

from endaq.plot import rolling_min_max_envelope
from endaq.calc import filters, integrate, utils, shock, psd


def shock_vibe_metrics(
        df: pd.DataFrame,
        tukey_percent: float = 0.1,
        highpass_cutoff: Optional[float] = None,
        accel_units: str = "gravity",
        disp_units: str = "in",
        freq_splits: typing.Union[np.ndarray, list, tuple] = (0, 65, 300, 1500, None),
        detrend: typing.Literal["start", "mean", "median", None] = "median",
        zero: typing.Literal["start", "mean", "median"] = "start",
        include_integration: bool = True,
        include_pseudo_velocity: bool = False,
        damp: float = 0.05,
        init_freq: float = 1.0,
        bins_per_octave: float = 12,
        include_resultant: bool = False,
        display_plots: bool = False,
) -> pd.DataFrame:
    """
    Compute the following shock and vibration metrics for a given time series dataframe
        - Peak Absolute Acceleration
        - RMS Acceleration
        - Peak Frequency
        - RMS Acceleration in Defined Frequency Ranges with `freq_splits`
        - Peak Absolute Velocity
        - RMS Velocity
        - Peak Absolute Displacement
        - RMS Displacement
        - Peak Pseudo Velocity & Corresponding Frequency
        
    :param df: the input dataframe with an index defining the time in seconds or datetime and units of `accel_units`
    :param tukey_percent: the portion of the time series to apply a Tukey window (a taper that forces beginning and end
        to 0), default is 0.1

        *  Note that the RMS metrics will only be computed on the portion of time that isn't tapered
    :param highpass_cutoff: the cutoff frequency of a preconditioning highpass
        filter; if None, no filter is applied. For shock events, it is recommended to set this to None (the default),
        but it is recommended for vibration.
    :param accel_units: the units to display acceleration as, default is `"gravity"` which will be shortened to 'g'
        in labels, the unit conversion is handled using :py:func:`~endaq.calc.utils.convert_units()`
    :param disp_units: the units to display displacement as and velocity (divided by seconds), default is `"in"`,
        the unit conversion is handled using :py:func:`~endaq.calc.utils.convert_units()`
    :param freq_splits: the boundaries of the frequency bins for the RMS calculations; must be strictly increasing, if
        `None` is given for the last value (the default) it will set this as the sampling rate
    :param detrend: the output quantity driven to zero prior to the calculations

        *  `None` does nothing
        *  `"start"` forces the first datapoint to 0,
        *  `"mean"` chooses ``-np.mean(output)``
        *  `"median"` (default) chooses ``-np.median(output)``
    :param zero: the output quantity driven to zero inside each integration call

        *  `"start"` (default) forces the first datapoint to 0,
        *  `"mean"` chooses ``-np.mean(output)``
        *  `"median"` (default) chooses ``-np.median(output)``
    :param include_integration: if `True`, include the calculations of velocity and displacement.  Defaults to `True`.
    :param include_pseudo_velocity: if `True`, include the more time-consuming calculation of pseudo velocity.
        Defaults to `False`.
    :param damp: the damping coefficient used in the shock response calculation `ζ`, related to the Q-factor by
        `ζ = 1/(2Q)`; defaults to 0.05
    :param init_freq: the initial frequency in the sequence for the shock response calculation; if `None`,
        use the frequency corresponding to the data's duration, default is 1.0 Hz
    :param bins_per_octave: the number of frequencies per octave for the shock response calculation
    :param include_resultant: add a resultant (root sum of the squares) for each metric,
        calculated from the other input dataframe columns
    :param display_plots: display plotly figures of the min/max envelope of acceleration, velocity, displacement and
        PVSS (default as False)
    :return: a dataframe containing all the metrics, one computed per column of the input dataframe

    Here is an example calculating and displaying these metrics for the bearing dataset discussed in our blog
    `Top 12 Vibration Metrics to Monitor & How to Calculate Them <https://blog.endaq.com/top-vibration-metrics-to-monitor-how-to-calculate-them>`_

    .. code:: python

        import endaq
        endaq.plot.utilities.set_theme('endaq_light')
        import pandas as pd
        import plotly.express as px

        # Get Acceleration Data
        accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

        # Calculate Metrics
        metrics = endaq.calc.stats.shock_vibe_metrics(accel, include_resultant=False, freq_splits=[0, 65, 300, None])

        # Generate Figure with Bar Plots
        fig = px.bar(
            metrics,
            x="variable", y="value", color='variable',
            hover_data = ["units"],
            facet_col="calculation", facet_col_wrap=3)
        fig.update_yaxes(matches=None, visible=False)
        fig.update_xaxes(visible=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.show()

    .. plotly::
       :fig-vars: fig

        import endaq
        endaq.plot.utilities.set_theme('endaq_light')
        import pandas as pd
        import plotly.express as px

        # Get Acceleration Data
        accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

        # Calculate Metrics
        metrics = endaq.calc.stats.shock_vibe_metrics(accel, include_resultant=False, freq_splits=[0, 65, 300, None])

        # Generate Figure with Bar Plots
        fig = px.bar(
            metrics,
            x="variable", y="value", color='variable',
            hover_data = ["units"],
            facet_col="calculation", facet_col_wrap=3)
        fig.update_yaxes(matches=None, visible=False)
        fig.update_xaxes(visible=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.show()

    """
    freq_splits = np.array(freq_splits)

    # Get unit conversion
    accel_2_disp = utils.convert_units(units_in=accel_units, units_out=disp_units + '/s^2')
    if accel_units == 'gravity':
        accel_units = 'g'

    # Remove Offset
    if detrend == "start":
        df -= df.iloc[0]
    elif detrend == "mean":
        df -= df.mean(axis="index")
    elif detrend == "median":
        df -= df.median(axis="index")
    elif detrend is not None:
        raise ValueError(f'kwarg detrend was {detrend}, must be one of [None, "start", "mean", "median"]')

    # Apply Filters & Window
    df = filters.butterworth(df, low_cutoff=highpass_cutoff, tukey_percent=tukey_percent)

    # Integrate
    data_list = [df]
    if include_integration:
        [accel, vel, disp] = integrate.integrals(
            df * accel_2_disp, n=2, tukey_percent=tukey_percent, highpass_cutoff=highpass_cutoff, zero=zero)
        data_list = [df, vel, disp]

    # Calculate Peak & RMS
    metrics = pd.DataFrame()
    rms_start = int(tukey_percent / 2 * df.shape[0])
    rms_end = df.shape[0] - rms_start
    for label, units, data in zip(
            ['Acceleration', 'Velocity', 'Displacement'],
            [accel_units, disp_units + '/s', disp_units],
            data_list):

        # Display Plots
        if display_plots:
            rolling_min_max_envelope(data, plot_as_bars=True, opacity=0.7).update_layout(yaxis_title_text=label).show()

        # Calculate Absolute Peak
        peak = pd.DataFrame(data.abs().max()).reset_index()
        peak.columns = ['variable', 'value']
        if include_resultant:
            peak = pd.concat([peak, pd.DataFrame({
                'variable': ['Resultant'],
                'value': [data.pow(2).sum(axis=1).pow(0.5).abs().max()]
            })])
        peak['calculation'] = 'Peak Absolute ' + label
        peak['units'] = units

        # Calculate RMS
        rms_stats = pd.DataFrame(data.iloc[rms_start:rms_end].pow(2).mean() ** 0.5).reset_index()
        rms_stats.columns = ['variable', 'value']
        if include_resultant:
            rms_stats = pd.concat([rms_stats, pd.DataFrame({
                'variable': ['Resultant'],
                'value': [np.sum(rms_stats.value ** 2) ** 0.5]
            })])
        rms_stats['calculation'] = f'RMS {label}'
        rms_stats['units'] = units

        # Add to Metrics
        metrics = pd.concat([metrics, peak, rms_stats])

    # Peak Frequency
    fs = 1 / utils.sample_spacing(df)
    parseval = psd.welch(df, scaling='parseval', bin_width=fs / len(df))
    peak = parseval.idxmax().reset_index()
    peak.columns = ['variable', 'value']
    peak['calculation'] = 'Peak Frequency'
    peak['units'] = 'Hz'
    metrics = pd.concat([metrics, peak])

    # Create Labels for Frequency Range Splits
    freq_splits = np.array(freq_splits)
    if freq_splits[-1] is None:
        freq_splits[-1] = int(np.floor(fs))
    freq_splits = freq_splits[freq_splits <= fs]
    labels = np.array(
        ['RMS from ' + str(freq_splits[i]) + ' to ' + str(freq_splits[i + 1]) for i in range(len(freq_splits) - 1)])

    # Calculate RMS in Defined Frequency Ranges
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*empty.*')
        rms_psd_breaks = psd.to_jagged(parseval, freq_splits, agg='sum') ** 0.5
    rms_psd_breaks.index = pd.Series(labels, name='calculation')
    if include_resultant:
        rms_psd_breaks['Resultant'] = rms_psd_breaks.sum(axis=1)
    rms_psd_breaks = rms_psd_breaks.reset_index().melt(id_vars='calculation')
    rms_psd_breaks['units'] = accel_units
    metrics = pd.concat([metrics, rms_psd_breaks])

    # Optionally Calculate PVSS
    if include_pseudo_velocity:
        freqs = utils.logfreqs(df, bins_per_octave=bins_per_octave, init_freq=init_freq)
        pvss = shock.shock_spectrum(df * accel_2_disp, freqs=freqs, damp=damp, mode='pvss')
        if include_resultant:
            pvss['Resultant'] = shock.shock_spectrum(df * accel_2_disp, freqs=freqs, damp=damp, mode='pvss',
                                                     aggregate_axes=True)['Resultant']

        if display_plots:
            px.line(pvss, log_x=True, log_y=True).update_layout(yaxis_title_text='Pseudo Velocity',
                                                                xaxis_title_text='Natural Frequency (Hz)').show()

        # Add Peak Value
        pvss_stats = pd.DataFrame(pvss.max()).reset_index()
        pvss_stats.columns = ['variable', 'value']
        pvss_stats['calculation'] = 'Peak Pseudo Velocity'
        pvss_stats['units'] = disp_units + '/s'
        metrics = pd.concat([metrics, pvss_stats])

        # Add Peak Frequency
        pvss_stats = pd.DataFrame(pvss.idxmax()).reset_index()
        pvss_stats.columns = ['variable', 'value']
        pvss_stats['calculation'] = 'Peak PVSS Frequency'
        pvss_stats['units'] = 'Hz'
        metrics = pd.concat([metrics, pvss_stats])

    # Return Metrics
    return metrics


def find_peaks(
        df: pd.DataFrame,
        time_distance: float = 1.0,
        add_resultant: bool = False,
        threshold: float = None,
        threshold_reference: typing.Literal["rms", "peak"] = "peak",
        threshold_multiplier: float = 0.1,
        use_abs: bool = True,
        display_plots: bool = False,
) -> pd.DataFrame:
    """
    Find the peak events of a given time series using the maximum across all input columns
    
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param time_distance: the minimum time in seconds between events, default is 1.0
    :param add_resultant: add a resultant (root sum of the squares) to the time series prior to finding peaks,
        calculated from the other input dataframe columns
    :param threshold: if `None` (default) this is ignored, but if defined this value is passed as the minimum threshold
        to define a shock event
    :param threshold_reference: if the threshold isn't defined, calculate it from:

        *  `"peak"` (default) the overall peak
        *  `"rms"` the RMS value of the time series
    :param threshold_multiplier: if the threshold isn't defined, multiply this by the `threshold_reference`,
        suggestions are:
        *  `0.1` (default) when using peak, to get all events greater than 10% of the overall peak
        *  `4.0` when using RMS, a typical Gaussian signal has a kurtosis of 3.0
    :param use_abs: use the absolute value of the data to define peak events, default is True
    :param display_plots: display a plotly figure of the time series with the peak events plotted over it (default as
        False)
    :return: an array of index locations

    .. seealso::
        - `SciPy find_peaks function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
          Documentation for the SciPy function used to identify the peak events.

    Here's an example implementation using a 60M dataset that loads the data, finds the peaks, and plots with the peak
    events identified all very quickly
    
    .. code:: python

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.graph_objects as go

        #Get Accel
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',measurement_type='accel',
            time_mode='datetime')

        #Filter
        accel = endaq.calc.filters.butterworth(accel,low_cutoff=2)

        #Get Peak Indexes
        indexes = endaq.calc.stats.find_peaks(
            accel, time_distance=2,
            threshold_reference="rms", threshold_multiplier=5.0)

        #Generate a Dataframe with Just the Peak Events
        df_peaks = accel.iloc[indexes]

        #Generate Shaded Bar Plot of All Data
        fig = endaq.plot.rolling_min_max_envelope(accel, plot_as_bars=True, opacity=0.7)

        #Add Peaks & Display
        fig.add_trace(
            go.Scatter(
                x=df_peaks.index,
                y=df_peaks.abs().max(axis=1).to_numpy(),
                mode='markers', name='Peak Events', marker_symbol='x', marker_color='white'
                )
            )
        fig.show()

    .. plotly::
       :fig-vars: fig

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.graph_objects as go

        #Get Accel
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',measurement_type='accel',
            time_mode='datetime')

        #Filter
        accel = endaq.calc.filters.butterworth(accel,low_cutoff=2)

        #Get Peak Indexes
        indexes = endaq.calc.stats.find_peaks(
            accel, time_distance=2,
            threshold_reference="rms", threshold_multiplier=5.0)

        #Generate a Dataframe with Just the Peak Events
        df_peaks = accel.iloc[indexes]

        #Generate Shaded Bar Plot of All Data
        fig = endaq.plot.rolling_min_max_envelope(accel, plot_as_bars=True, opacity=0.7)

        #Add Peaks & Display
        fig.add_trace(
            go.Scatter(
                x=df_peaks.index,
                y=df_peaks.abs().max(axis=1).to_numpy(),
                mode='markers', name='Peak Events', marker_symbol='x', marker_color='white'
                )
            )
        fig.show()
    """
    df_unmodified = df.copy()

    # Add Resultant
    if add_resultant:
        df['Resultant'] = df.pow(2).sum(axis=1).pow(0.5)

    # Reduce to Only Maximum Values Per Row
    if use_abs:
        peaks = df.abs().max(axis=1).to_numpy()
    else:
        peaks = df.max(axis=1).to_numpy()

    # Define Threshold
    if threshold is None:
        if threshold_reference == 'peak':
            threshold = np.max(peaks) * threshold_multiplier
        elif threshold_reference == 'rms':
            rms_val = df_unmodified.pow(2).mean() ** 0.5
            threshold = np.max(rms_val) * threshold_multiplier
            if add_resultant:
                threshold = np.sum(rms_val ** 2) ** 0.5 * threshold_multiplier

    # Find Peak Indexes
    indexes = signal.find_peaks(
        peaks,
        distance=int(time_distance / utils.sample_spacing(df)),
        height=threshold,
    )[0]

    # Optionally Display Plot
    if display_plots:
        df_peaks = df.iloc[indexes]
        fig = rolling_min_max_envelope(df, plot_as_bars=True, opacity=0.7)
        fig.add_trace(go.Scattergl(
            x=df_peaks.index,
            y=df_peaks.abs().max(axis=1).to_numpy(),
            mode='markers',
            name='Peak Events'
        ))
        fig.show()

    # Return Peak Times
    return indexes


def rolling_metrics(
        df: pd.DataFrame,
        indexes: np.array = None,
        index_values: np.array = None,
        num_slices: int = 100,
        slice_width: float = None,
        **kwargs,
) -> pd.DataFrame:
    """
    Quantify a series of time slices of a given time series
    
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param indexes: the index locations (not value) of each peak event to quantify like what is returned by :py:func:`~endaq.calc.stats.find_peaks()`
    :param index_values: the index values of each peak event to quantify (slower but more intuitive than using `indexes`)
    :param num_slices: the number of slices to split the time series into, default is 100,
        this is ignored if `indexes` or `index_values` are defined
    :param slice_width: the time in seconds to center about each slice index,
        if none is provided it will calculate one based upon the number of slices
    :param kwargs: Other parameters to pass directly to :py:func:`~endaq.calc.stats.shock_vibe_metrics()`
    :return: a dataframe containing all the metrics, one computed per column of the input dataframe, and one per peak event

    Here's a continuation of the example shown in :py:func:`~endaq.calc.stats.find_peaks()` that generates a table of
    metrics for a few defined time stamps, and then a row of subplots for each metric calculated.

    .. code:: python

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.express as px
        import pandas as pd

        # Get Accel
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',measurement_type='accel',
            time_mode='datetime')

        # Filter
        accel = endaq.calc.filters.butterworth(accel,low_cutoff=2)

        # Calculate for 3 Specific Times
        metrics = endaq.calc.stats.rolling_metrics(
            accel,
            index_values = pd.DatetimeIndex(['2020-03-13 23:40:13', '2020-03-13 23:45:00', '2020-03-13 23:50:00'],tz='UTC'),
            slice_width=5.0)

        # Simplify Timestamp Column
        metrics.timestamp = metrics.timestamp.astype(str).map(lambda x: x[10:19])

        # Generate Plot Table of Metrics
        table_plot = endaq.plot.table_plot(metrics)
        table_plot.show()

        # Calculate for 50 Equally Spaced & Sized Slices, Turning off Pseudo Velocity (Only Recommended for Smaller Time Slices)
        metrics = endaq.calc.stats.rolling_metrics(
            accel, num_slices=50, highpass_cutoff=2,
            tukey_percent=0.0, include_pseudo_velocity=False)


        # Generate Row with Subplots for each metric
        metrics_fig = px.scatter(
            metrics,
            x='timestamp',
            y='value',
            color='variable',
            facet_col='calculation',
            facet_col_spacing=0.03
        )
        metrics_fig.update_yaxes(title_text='', matches=None, showticklabels=True).update_xaxes(title_text='')
        metrics_fig.update_layout(width=3000, legend_y=1.2, legend_title_text='').for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        metrics_fig.show()

    .. plotly::
       :fig-vars: table_plot, metrics_fig

        import endaq
        endaq.plot.utilities.set_theme()
        import plotly.express as px
        import pandas as pd

        # Get Accel
        accel = endaq.ide.get_primary_sensor_data('https://info.endaq.com/hubfs/ford_f150.ide',measurement_type='accel',
            time_mode='datetime')

        # Filter
        accel = endaq.calc.filters.butterworth(accel,low_cutoff=2)

        # Calculate for 3 Specific Times
        metrics = endaq.calc.stats.rolling_metrics(
            accel,
            index_values = pd.DatetimeIndex(['2020-03-13 23:40:13', '2020-03-13 23:45:00', '2020-03-13 23:50:00'],tz='UTC'),
            slice_width=5.0)

        # Simplify Timestamp Column
        metrics.timestamp = metrics.timestamp.astype(str).map(lambda x: x[10:19])

        # Generate Plot Table of Metrics
        table_plot = endaq.plot.table_plot(metrics)
        table_plot.show()

        # Calculate for 50 Equally Spaced & Sized Slices, Turning off Pseudo Velocity (Only Recommended for Smaller Time Slices)
        metrics = endaq.calc.stats.rolling_metrics(
            accel, num_slices=50, highpass_cutoff=2,
            tukey_percent=0.0, include_pseudo_velocity=False)


        # Generate Row with Subplots for each metric
        metrics_fig = px.scatter(
            metrics,
            x='timestamp',
            y='value',
            color='variable',
            facet_col='calculation',
            facet_col_spacing=0.03
        )
        metrics_fig.update_yaxes(title_text='', matches=None, showticklabels=True).update_xaxes(title_text='')
        metrics_fig.update_layout(width=3000, legend_y=1.2, legend_title_text='').for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        metrics_fig.show()

    """
    indexes, slice_width, num, length = utils._rolling_slice_definitions(
        df,
        indexes=indexes,
        index_values=index_values,
        num_slices=num_slices,
        slice_width=slice_width
    )

    # Loop through and compute metrics
    metrics = pd.DataFrame()
    for i in indexes:
        window_start = max(0, i - num)
        window_end = min(length, i + num)
        event_metrics = shock_vibe_metrics(
            df.iloc[window_start:window_end],
            **kwargs
        )
        event_metrics['timestamp'] = df.index[i]
        metrics = pd.concat([metrics, event_metrics])

    return metrics


def L2_norm(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the L2 norm (a.k.a. the Euclidean Norm).

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the L2 norm is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions of size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.sum(np.abs(array) ** 2, axis=axis, keepdims=keepdims))


def max_abs(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Compute the maximum of the absolute value of an array.

    This function should be equivalent to, but generally use less memory than
    ``np.amax(np.abs(array))``.

    Specifically, it generates the absolute-value maximum from ``np.amax(array)``
    and ``-np.amin(array)``. Thus instead of allocating space for the
    intermediate array ``np.abs(array)``, it allocates for the axis-collapsed
    smaller arrays ``np.amax(array)`` & ``np.amin(array)``.

    .. note:: This method does **not** work on complex-valued arrays.

    :param array: the input data
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the absolute maximum is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    # Forbid complex-valued data
    if np.iscomplexobj(array):
        raise ValueError("`max_abs` does not accept complex arrays")

    return np.maximum(
        np.amax(array, initial=-np.inf, axis=axis, keepdims=keepdims),
        -np.amin(array, initial=np.inf, axis=axis, keepdims=keepdims),
    )


def rms(
    array: np.ndarray,
    axis: Union[None, typing.SupportsIndex, Sequence[typing.SupportsIndex]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    """
    Calculate the root-mean-square (RMS) along a given axis.

    :param array: the input array
    :param axis: the axis/axes along which to aggregate; if `None` (default),
        the RMS is computed along the flattened array
    :param keepdims: if `True`, the axes which are reduced are left in the
        result as dimensions with size one; if `False` (default), the reduced
        axes are removed
    :return: an array containing the computed values
    """
    return np.sqrt(np.mean(np.abs(array) ** 2, axis=axis, keepdims=keepdims))


def rolling_rms(
    df: Union[pd.DataFrame, pd.Series], window_len: int, *args, **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate a rolling root-mean-square (RMS) over a pandas `DataFrame`.

    This function is equivalent to, but computationally faster than the following::

        df.rolling(window_len).apply(endaq.calc.stats.rms)

    :param df: the input data
    :param window_len: the length of the rolling window
    :param args: the positional arguments to pass into ``df.rolling().mean``
    :param kwargs: the keyword arguments to pass into ``df.rolling().mean``
    :return: the rolling-windowed RMS

    .. seealso::
        
        - `Pandas Rolling Mean <https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.mean.html>`_
          - official documentation for ``df.rolling().mean``
        - `Pandas Rolling Standard Deviation method <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.std.html>`_
          - similar to this function, but first removes the windowed mean before squaring
    """
    return df.pow(2).rolling(window_len).mean(*args, **kwargs).apply(np.sqrt, raw=True)
