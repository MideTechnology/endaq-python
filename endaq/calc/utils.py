# -*- coding: utf-8 -*-

from __future__ import annotations

import typing
from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd
import scipy.signal
import pint


def sample_spacing(
    data: Union[np.ndarray, pd.DataFrame],
    convert: typing.Literal[None, "to_seconds"] = "to_seconds",
) -> Union[None, float, np.timedelta64]:
    """
    Calculate the average spacing between individual samples.

    For time indices, this calculates the sampling period `dt`.

    :param data: the input data; either a pandas DataFrame with the samples
        spaced along its index, or a 1D-array-like of sample times
    :param convert: if `"to_seconds"` (default), convert any time objects into
        floating-point seconds
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.index
    if len(data) <= 1:
        return None

    dt = (data[-1] - data[0]) / (len(data) - 1)
    if convert == "to_seconds" and isinstance(dt, (np.timedelta64, pd.Timedelta)):
        dt = dt / np.timedelta64(1, "s")

    return dt


def logfreqs(
    df: pd.DataFrame, init_freq: Optional[float] = None, bins_per_octave: float = 12.0
) -> np.ndarray:
    """
    Calculate a sequence of log-spaced frequencies for a given dataframe.

    :param df: the input data
    :param init_freq: the initial frequency in the sequence; if `None` (default),
        use the frequency corresponding to the data's duration
    :param bins_per_octave: the number of frequencies per octave
    :return: an array of log-spaced frequencies
    """
    dt = sample_spacing(df)
    T = dt * len(df.index)

    if init_freq is None:
        init_freq = 1 / T
    elif 1 / init_freq > T:
        warnings.warn(
            "the data's duration is too short to accurately represent an"
            f" initial frequency of {init_freq:.3f} Hz",
            RuntimeWarning,
        )

    return 2 ** np.arange(
        np.log2(init_freq),
        np.log2(1 / dt) - 1,
        1 / bins_per_octave,
    )


def to_dB(
    data: np.ndarray,
    reference: Union[float, typing.Literal[tuple(dB_refs.keys())]],
    squared: bool = False,
) -> np.ndarray:
    """
    Scale data into units of decibels.

    Decibels are a log-scaled ratio of some value against a reference;
    typically this is expressed as follows:

    .. math:: dB = 10 \\log_{10}\\left( \\frac{x}{x_{\\text{ref}}} \\right)

    By convention, "decibel" units tend to operate on units of *power*. For
    units that are proportional to power *when squared* (e.g., volts, amps,
    pressure, etc.), their "decibel" representation is typically doubled (i.e.,
    :math:`dB = 20 \\log_{10}(...)`). Users can specify which scaling to use
    with the `squared` parameter.

    .. note::
        Decibels can **NOT** be calculated from negative values.

        For example, to calculate dB on arbitrary time-series data, typically
        data is first aggregated via:

        - a *total RMS* (like :py:func:`endaq.calc.stats.rms`),
        - a *rolling RMS* (like :py:func:`endaq.calc.stats.rolling_rms`), or
        - a *PSD* (like :py:func:`endaq.calc.psd.welch`),

        and the non-negative result from the aggregation is then scaled into
        decibels.

    :param data: the input data
    :param reference: the reference value corresponding to 0dB
    :param squared: whether the input data & reference value are pre-squared;
        defaults to `False`
    """
    if isinstance(reference, str):
        try:
            reference = dB_refs[reference]
        except KeyError:
            raise ValueError(f'unknown reference "{reference}"')
    elif reference <= 0:
        raise ValueError("reference value must be strictly positive")

    data = np.asarray(data)
    if np.any(data < 0):
        raise ValueError(
            "cannot compute decibels from negative values (see the docstring"
            " for details)"
        )

    return (10 if squared else 20) * (np.log10(data) - np.log10(reference))


dB_refs = {
    "SPL": 2e-5,  # Pascal
    "audio_intensity": 1e-12,  # W/m²
}


def resample(df: pd.DataFrame, sample_rate: Optional[float] = None) -> pd.DataFrame:
    """
    Resample a dataframe to a desired sample rate (in Hz)

    :param df: The DataFrame to resample, indexed by time
    :param sample_rate: The desired sample rate to resample the given data to.
     If one is not supplied, then it will use the same as it currently does, but
     make the time stamps uniformly spaced
    :return: The resampled data in a DataFrame
    """
    if sample_rate is None:
        num_samples_after_resampling = len(df)
    else:
        dt = sample_spacing(df)
        num_samples_after_resampling = int(dt * len(df) * sample_rate)

    resampled_data, resampled_time = scipy.signal.resample(
        df,
        num_samples_after_resampling,
        t=df.index.values.astype(np.float64),
    )

    # Check for datetimes, if so localize
    if 'datetime' in str(df.index.dtype):
        df.index = df.index.tz_localize(None)

    resampled_df = pd.DataFrame(
        resampled_data,
        index=resampled_time.astype(df.index.dtype),
        columns=df.columns,
    )
    
    resampled_df.index.name = df.index.name
    
    return resampled_df


def _rolling_slice_definitions(
        df: pd.DataFrame,
        num_slices: int = 100,
        indexes: np.array = None,
        index_values: np.array = None,
        slice_width: float = None,
):
    """
    Compute parameters needed to define index locations and slice width for rolling computations

    :param df: the input dataframe with an index defining the time in seconds or datetime
    :param num_slices: the number of slices to split the time series into, default is 100,
        this is ignored if `indexes` is defined
    :param indexes: the center index locations (not value) of each slice to compute the FFT
    :param index_values: the index values of each peak event to quantify (slower but more intuitive than using `indexes`)
    :param slice_width: the time in seconds to center about each slice index,
        if none is provided it will calculate one based upon the number of slices
    :return: a tuple of `indexes`, `slice_width`, `num`, and `length`

    See example use cases and syntax at :py:func:`~endaq.plot.spectrum_over_time()`
    which visualizes the output of this function in Heatmaps, Waterfall plots,
    Surface plots, and Animations

    """

    length = len(df)

    # Define center index locations of each slice if not provided
    if indexes is None:
        if index_values is not None:
            indexes = np.zeros(len(index_values), int)
            for i in range(len(indexes)):
                indexes[i] = int((np.abs(df.index - index_values[i])).argmin())
        else:
            indexes = np.linspace(0, length, num_slices, endpoint=False, dtype=int)
            indexes = indexes + int(indexes[1] / 2)

    # Calculate slice step size
    spacing = sample_spacing(df)
    if slice_width is None:
        slice_width = spacing * length / len(indexes)
    num = int(slice_width / spacing / 2)

    return indexes, slice_width, num, length


def convert_units(
        units_in: str,
        units_out: str,
        df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Using the `Pint library <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_ apply a unit
    conversion to a provided unit-unaware dataframe.

    :param units_in: a text string defining the base units to convert from like `"in"` for inches
    :param units_out: a text string defining the destination units to convert to like `"mm"` for millimeters
    :param df: the input dataframe, if none the unit conversion is only applied from `units_in` to `units_out`
    :return: a dataframe with the values scaled according to the unit conversion, if no dataframe is provided then a
        scaler value is returned

    Some examples are provided below which includes a table of common unit conversions. A full list is available from
    the `Pint library <https://github.com/hgrecco/pint/blob/master/pint/default_en.txt>`_.

    .. code:: python3

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd
        import plotly.express as px

        # Simple conversion factor from inches to millimeters
        in_2_mm = endaq.calc.utils.convert_units('in', 'mm')

        # Get idelib dataset
        doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/All-Channels.ide')

        # Get acceleration data in 'g' and convert to in/s^2
        accel_in_gs = endaq.ide.get_primary_sensor_data(doc=doc, measurement_type='accel')
        accel_in_inches = endaq.calc.utils.convert_units('gravity', 'in/s**2', accel_in_gs)

        # Get temperature in Celsius and convert to Fahrenheit
        temp_in_C = endaq.ide.get_primary_sensor_data(doc=doc, measurement_type='temp')
        temp_in_F = endaq.calc.utils.convert_units('degC', 'degF', temp_in_C)

        # Merge C & F in one dataframe
        temp_in_C.columns = ['Temperature in Degrees C']
        temp_in_F.columns = ['Temperature in Degrees F']
        temp = pd.concat([temp_in_C, temp_in_F], axis=1)

        # Display plot with both C and F
        fig = px.line(
            temp.reset_index().melt(id_vars='timestamp'),
            x='timestamp',
            y='value',
            facet_col='variable',
            facet_col_spacing=0.07
        ).for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(matches=None, showticklabels=True, title_text='').update_xaxes(title_text='')
        fig.show()

        # Get Table of Unit Conversions
        df = pd.read_csv('https://info.endaq.com/hubfs/Unit-Conversion-Examples.csv')
        df['output'] = 0
        for i in df.index:
            df.loc[i, 'output'] = endaq.calc.utils.convert_units(
                units_in = df.loc[i, 'units_in'],
                units_out = df.loc[i, 'units_out'])

        # Generate Plot Table
        plot_table = endaq.plot.table_plot(df, num_round=6)
        plot_table.show()

    .. plotly::
        :fig-vars: fig, plot_table

        import endaq
        endaq.plot.utilities.set_theme()
        import pandas as pd
        import plotly.express as px

        # Simple conversion factor from inches to millimeters
        in_2_mm = endaq.calc.utils.convert_units('in', 'mm')

        # Get idelib dataset
        doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/All-Channels.ide')

        # Get acceleration data in 'g' and convert to in/s^2
        accel_in_gs = endaq.ide.get_primary_sensor_data(doc=doc, measurement_type='accel')
        accel_in_inches = endaq.calc.utils.convert_units('gravity', 'in/s**2', accel_in_gs)

        # Get temperature in Celsius and convert to Fahrenheit
        temp_in_C = endaq.ide.get_primary_sensor_data(doc=doc, measurement_type='temp')
        temp_in_F = endaq.calc.utils.convert_units('degC', 'degF', temp_in_C)

        # Merge C & F in one dataframe
        temp_in_C.columns = ['Temperature in Degrees C']
        temp_in_F.columns = ['Temperature in Degrees F']
        temp = pd.concat([temp_in_C, temp_in_F], axis=1)

        # Display plot with both C and F
        fig = px.line(
            temp.reset_index().melt(id_vars='timestamp'),
            x='timestamp',
            y='value',
            facet_col='variable',
            facet_col_spacing=0.07
        ).for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(matches=None, showticklabels=True, title_text='').update_xaxes(title_text='')
        fig.show()

        # Get Table of Unit Conversions
        df = pd.read_csv('https://info.endaq.com/hubfs/Unit-Conversion-Examples.csv')
        df['output'] = 0
        for i in df.index:
            df.loc[i, 'output'] = endaq.calc.utils.convert_units(
                units_in = df.loc[i, 'units_in'],
                units_out = df.loc[i, 'units_out'])

        # Generate Plot Table
        plot_table = endaq.plot.table_plot(df, num_round=6)
        plot_table.show()

    """
    ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)
    src = ureg(units_in)
    dst = ureg(units_out)

    if df is None:
        return src.to(dst).magnitude
    else:
        converted_df = df.copy()
        vals = (df.values * src).to(dst).magnitude
        for i, c in enumerate(df.columns):
            converted_df[c] = vals[:, i]
        return converted_df
