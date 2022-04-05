from __future__ import annotations

import typing  # for `SupportsIndex`, which is Python3.8+ only
from typing import Union
from collections.abc import Sequence

import numpy as np
import pandas as pd

from endaq.calc import filters, integrate, shock, sample_spacing, logfreqs
from endaq.plot import rolling_min_max_envelope


def shock_vibe_metrics(
        df: pd.DataFrame,
        tukey_percent: float = 0.1,
        highpass_cutoff: float = None,
        vel_disp_multiplier: float = 1.0,
        zero: typing.Literal["start", "mean", "median", None] = "median",
        include_integration: bool = True,
        include_pseudo_velocity: bool = True,
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
        - Peak Absolute Velocity
        - RMS Velocity
        - Peak Absolute Displacement
        - RMS Displacement
        - Peak Pseudo Velocity
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param tukey_percent: the portion of the time series to apply a Tukey window (a taper that forces beginning and end to 0), default is 0.1
        - Note that the RMS metrics will only be computed on the portion of time that isn't tapered
    :param zero: the output quantity driven to zero prior to the calculations
        `None` does nothing
        `"start"` forces the first datapoint to 0,
        `"mean"` chooses ``-np.mean(output)`` & `"median"` (default) chooses
        ``-np.median(output)``
    :param highpass_cutoff: the cutoff frequency of a preconditioning highpass
        filter; if None, no filter is applied. For shock events, it is recommended to set this to None (the default), but it is recommended for vibration.
    :param vel_disp_multiplier: applies a scale to the velocity and displacement metrics
        - 386.09 to convert from g to inches (in)
        - 9806.65 to convert from g to millimeters (mm)
    :param include_integration: include the calculations of velocity and displacement, default behavior
    :param include_pseudo_velocity: include the more time consuming calculation of pseudo velocity, default behavior
    :param damp: the damping coefficient `ζ`, related to the Q-factor by
        `ζ = 1/(2Q)`; defaults to 0
    :param init_freq: the initial frequency in the sequence; if `None`,
        use the frequency corresponding to the data's duration, default is 1.0 Hz
    :param bins_per_octave: the number of frequencies per octave
    :param include_resultant: add a resultant (root sum of the squares) for each metric, calculated from the other input dataframe columns
    :param display_plots: display plotly figures of the min/max envelope of acceleration, velocity, displacement and PVSS (default as False)
    :return: a dataframe containing all the metrics, one computed per column of the input dataframe
    """

    #Remove Offset
    if zero is not None:
        if zero == "start":
            df = df - df.iloc[0]
        elif zero in ("mean", "median"):
            zero = dict(mean=np.mean, median=np.median)[zero]
            df = df - df.apply(zero, axis="index", raw=True)

    #Apply Filters & Window
    df = filters.butterworth(df, low_cutoff=highpass_cutoff, tukey_percent=tukey_percent)

    #Integrate
    data_list = [df]
    if include_integration:
        [accel, vel, disp] = integrate.integrals(
            df, n=2, tukey_percent=tukey_percent, highpass_cutoff=highpass_cutoff)
        data_list = [df, vel, disp]

    #Calculate Peak & RMS
    metrics = pd.DataFrame()
    rms_start = int(tukey_percent/2*df.shape[0])
    rms_end = df.shape[0] - rms_start
    for (label,data,scale) in zip(
            ['Acceleration','Velocity','Displacement'],
            data_list,
            [1,vel_disp_multiplier,vel_disp_multiplier]):

        #Apply Scale
        data = data * scale

        #Display Plots
        if display_plots:
            rolling_min_max_envelope(data).update_layout(yaxis_title_text = label).show()

        #Calculate Absolute Peak
        peak = pd.DataFrame(data.abs().max()).reset_index()
        peak.columns = ['variable','value']
        if include_resultant:
            peak = pd.concat([peak, pd.DataFrame({
                'variable':['Resultant'],
                'value':[np.sum(peak.value**2)**0.5]
            })])
        peak['calculation'] = 'Peak Absolute ' + label

        #Calculate RMS
        rms = pd.DataFrame(data.iloc[rms_start:rms_end].pow(2).mean()**0.5).reset_index()
        rms.columns = ['variable','value']
        if include_resultant:
            rms = pd.concat([rms, pd.DataFrame({
                'variable':['Resultant'],
                'value':[np.sum(rms.value**2)**0.5]
            })])
        rms['calculation'] = 'RMS ' + label

        #Add to Metrics
        metrics = pd.concat([metrics,peak,rms])

    #Optionally Calculate PVSS
    if include_pseudo_velocity:
        freqs = logfreqs(df, bins_per_octave=bins_per_octave, init_freq=init_freq)
        pvss = shock.shock_spectrum(df, freqs=freqs, damp=damp, mode='pvss')*vel_disp_multiplier
        if include_resultant:
            pvss['Resultant'] = shock.shock_spectrum(df, freqs=freqs, damp=damp, mode='pvss', aggregate_axes=True)['resultant']*vel_disp_multiplier

        if display_plots:
            px.line(pvss,log_x=True,log_y=True).update_layout(yaxis_title_text = 'Pseudo Velocity',xaxis_title_text='Natural Frequency (Hz)').show()

        pvss = pd.DataFrame(pvss.max()).reset_index()
        pvss.columns = ['variable','value']
        pvss['calculation'] = 'Peak Pseudo Velocity'
        metrics = pd.concat([metrics,pvss])

    #Return Metrics
    return metrics


def find_peaks(
        df: pd.DataFrame,
        time_distance: float = 1.0,
        add_resultant: bool = False,
        threshold: float = None,
        threshold_reference: typing.Literal["rms", "peak"] = "rms",
        threshold_multiplier: float = 4.0,
        use_abs: bool = True,
        display_plots: bool = False,
) -> pd.DataFrame:
    """
    Find the peak events of a given time series using the maximum across all input columns
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param time_distance: the minimum time in seconds between events, default is 1.0
    :param add_resultant: add a resultant (root sum of the squares) to the time series prior to finding peaks, calculated from the other input dataframe columns
    :param threshold: if `None` (default) this is ignored, but if defined this value is passed as the minimum threshold to define a shock event
    :param threshold_reference: if the threshold isn't defined, calculate it from:
        - `"rms"` (default) the RMS value of the time series
        - `"peak"` the overall peak
    :param threshold_multiplier: if the threshold isn't defined, multiply this by the `threshold_reference`, suggestions are:
        - `4.0` (default) when using RMS, a typical Gaussian signal has a kurtosis of 3.0
        - `0.1` when using peak, to get all events greater than 10% of the overall peak
    :param use_abs: use the absolute value of the data to define peak events, default is True
    :param display_plots: display a plotly figure of the time series with the peak events plotted over it (default as False)
    :return: a dataframe indexed by time containing only the peak events and their corresponding values

    .. seealso::
        - `SciPy find_peaks function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_
          Documentation for the SciPy function used to identify the peak events.
    """
    df_unmodified = df.copy()

    #Add Resultant
    if add_resultant:
        df['Resultant'] = df.pow(2).sum(axis=1).pow(0.5)

    #Reduce to Only Maximum Values Per Row
    if use_abs:
        peaks = df.abs().max(axis=1).to_numpy()
    else:
        peaks = df.max(axis=1).to_numpy()

    #Define Threshold
    if threshold is None:
        if threshold_reference=='peak':
            threshold = np.max(peaks) * threshold_multiplier
        elif threshold_reference=='rms':
            rms = df_unmodified.pow(2).mean()**0.5
            threshold = np.max(rms) * threshold_multiplier
            if add_resultant:
                threshold = np.sum(rms**2)**0.5 * threshold_multiplier

    #Find Peak Indexes
    indexes = scipy.signal.find_peaks(
        peaks,
        distance=int(time_distance/sample_spacing(df)),
        height=threshold,
    )[0]

    #Optionally Display Plot
    if display_plots:
        df_peaks = df.iloc[indexes]
        fig = rolling_min_max_envelope(df)
        fig.add_trace(go.Scattergl(
            x=df_peaks.index,
            y=df_peaks.abs().max(axis=1).to_numpy(),
            mode='markers',
            name='Peak Events'
        ))
        fig.show()

    #Return Peak Times
    return indexes


def quantify_peaks(
        df: pd.DataFrame,
        indexes: np.array,
        time_distance: float = 1.0,
        **kwargs,
) -> pd.DataFrame:
    """
    Quantify the peak events of a given time series, defined by their indexes
    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param indexes: the index locations of each peak event to quantify
    :param time_distance: the time in seconds to center about each peak index, default is 1.0
    :param vel_disp_multiplier: applies a scale to the velocity and displacement metrics
        - 386.09 to convert from g to inches (in)
        - 9806.65 to convert from g to millimeters (mm)
    :param kwargs: Other parameters to pass directly to :py:func:`shock_vibe_metrics`
    :return: a dataframe containing all the metrics, one computed per column of the input dataframe, and one per peak event
    """
    #Define number of points to step forward/back from each event
    num = int(time_distance / sample_spacing(df) / 2)
    length = len(df)

    #Loop through and compute metrics
    metrics = pd.DataFrame()
    for i in indexes:
        window_start = max(0, i - num)
        window_end = min(length, i + num)
        event_metrics = shock_vibe_metrics(
            df.iloc[window_start:window_end],
            **kwargs
        )
        event_metrics['timestamp'] = df.index[i]
        metrics = pd.concat([metrics,event_metrics])

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
