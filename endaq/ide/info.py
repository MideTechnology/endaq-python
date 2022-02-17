"""
Functions for retrieving summary data from a dataset.
"""
from __future__ import annotations
import typing

from collections import defaultdict
import datetime
import dateutil.tz
import warnings

import numpy as np
import pandas as pd
import pandas.io.formats.style
import idelib.dataset

from .measurement import MeasurementType, ANY, get_channels
from .files import get_doc
from .util import parse_time


__all__ = [
    "get_channel_table",
    "to_pandas",
    "get_primary_sensor_data",
]


# ============================================================================
# Display formatting functions
# ============================================================================

def format_channel_id(ch: idelib.dataset.Channel) -> str:
    """ Function for formatting an `idelib.dataset.Channel` or `SubChannel`
        for display. Renders as only the channel and subchannel IDs (the other
        information is shown in the rest of the table).

        :param ch: The `idelib.dataset.Channel` or `idelib.dataset.SubChannel`
            to format.
        :return: A formatted "channel.subchannel" string.
    """
    try:
        if ch.parent:
            return f"{ch.parent.id}.{ch.id}"
        else:
            return f"{ch.id}.*"

    except (AttributeError, TypeError, ValueError) as err:
        warnings.warn(f"format_channel_id({ch!r}) raised {type(err).__name__}: {err}")
        return str(ch)


def format_timedelta(val: typing.Union[int, float, datetime.datetime, datetime.timedelta]) -> str:
    """ Function for formatting microsecond timestamps (e.g., start, end,
        or duration) as times. Somewhat more condensed than the standard
        `DataFrame` formatting of `datetime.timedelta`.

        :param val: The `pandas.Timedelta` or `datetime.timedelta` to format.
            Will also work with microseconds as `float` or `int`.
        :return: A formatted time 'duration' string.
    """
    try:
        if isinstance(val, datetime.timedelta):
            td = pd.Timedelta(val)
        else:
            td = pd.Timedelta(microseconds=val)

        # NOTE: `components` attr only exists in pandas `Timedelta`
        c = td.components
        s = f"{c.minutes:02d}:{c.seconds:02d}.{c.milliseconds:04d}"
        if c.hours or c.days:
            s = f"{c.hours:02d}:{s}"
            if c.days:
                s = f"{c.days}d {s}"
        return s

    except (AttributeError, TypeError, ValueError) as err:
        warnings.warn(f"format_timedelta({val!r}) raised {type(err).__name__}: {err}")
        return str(val)


def format_timestamp(ts: typing.Union[int, float]) -> str:
    """ Function for formatting start/end timestamps. Somewhat more condensed
        than the standard Pandas formatting.

        :param ts: The timestamps in microseconds. Rendered as integers, since
            `idelib` timestamps have whole microsecond resolution.
        :return: A formatted timestamp string, with units.
    """
    try:
        return f"{int(ts)} µs"
    except (TypeError, ValueError) as err:
        warnings.warn(f"format_timestamp({ts!r}) raised {type(err).__name__}: {err}")
        return str(ts)

# ============================================================================
#
# ============================================================================


""" The default table formatting. """
TABLE_FORMAT = {
    'channel': format_channel_id,
    'start': format_timedelta,
    'end': format_timedelta,
    'duration': format_timedelta,
    'rate': "{:.2f} Hz",
}


def get_channel_table(dataset: typing.Union[idelib.dataset.Dataset, list],
                      measurement_type=ANY, 
                      start: typing.Union[int, float, str, datetime.datetime, datetime.timedelta] = 0,
                      end: typing.Optional[int, float, str, datetime.datetime, datetime.timedelta] = None,
                      formatting: typing.Optional[dict] = None,
                      index: bool = True, 
                      precision: int = 4,
                      timestamps: bool = False, 
                      **kwargs) -> typing.Union[pd.DataFrame, pd.io.formats.style.Styler]:
    """ Get summary data for all `SubChannel` objects in a `Dataset` that
        contain one or more type of sensor data. By using the optional
        `start` and `end` parameters, information can be retrieved for a
        specific interval of time.

        The `start` and `end` times, if used, may be specified in several
        ways:

        * `int`/`float` (Microseconds from the recording start)
        * `str` (formatted as a time from the recording start, e.g., `MM:SS`,
          `HH:MM:SS`, `DDd HH:MM:SS`). More examples:

          * ``":01"`` or ``":1"`` or ``"1s"`` (1 second)
          * ``"22:11"`` (22 minutes, 11 seconds)
          * ``"3:22:11"`` (3 hours, 22 minutes, 11 seconds)
          * ``"1d 3:22:11"`` (1 day, 3 hours, 22 minutes, 11 seconds)
        * `datetime.timedelta` or `pandas.Timedelta` (time from the
          recording start)
        * `datetime.datetime` (an explicit UTC time)

        :param dataset: A `idelib.dataset.Dataset` or a list of
            channels/subchannels from which to build the table.
        :param measurement_type: A :py:class:`~endaq.ide.MeasurementType`, a
            measurement type 'key' string, or a string of multiple keys
            generated by adding and/or subtracting
            :py:class:`~endaq.ide.MeasurementType` objects to filter the
            results. Any 'subtracted' types will be excluded.
        :param start: The starting time. Defaults to the start of the
            recording.
        :param end: The ending time. Defaults to the end of the recording.
        :param formatting: A dictionary of additional style/formatting items
            (see `pandas.DataFrame.style.format()`). If `False`, no additional
            formatting is applied.
        :param index: If `True`, show the index column on the left.
        :param precision: The default decimal precision to display. Can be
            changed later.
        :param timestamps: If `True`, show the start and end as raw
            microsecond timestamps.
        :returns: A table (`pandas.io.formats.style.Styler`) of summary data.
        :rtype: pandas.DataFrame
    """
    # We don't support multiple sessions on current Slam Stick/enDAQ recorders,
    # but in the event we ever do, this allows one to be specified like so:
    #       :param session: A `Session` or session ID to retrieve from a
    #           multi-session recording.
    # Leave out of docstring until we ever support it.
    session = kwargs.get('session', None)
    if session:
        session = getattr(session, 'sessionId', session)

    if hasattr(dataset, 'getPlots'):
        sources = get_channels(dataset, measurement_type)
    else:
        sources = dataset

    result = defaultdict(list)
    for source in sources:
        range_start = range_end = duration = rate = session_start = None
        samples = 0

        data = source.getSession(session)
        if data.session.utcStartTime:
            session_start = datetime.datetime.utcfromtimestamp(data.session.utcStartTime)
        start = parse_time(start, session_start)
        end = parse_time(end, session_start)

        if len(data):
            if not start and not end:
                start_idx, end_idx = 0, -1
                samples = len(data)
            else:
                start_idx, end_idx = data.getRangeIndices(start, end)
                end_idx = min(len(data) - 1, end_idx)
                if end_idx < 0:
                    samples = len(data) - start_idx - 1
                else:
                    samples = end_idx - start_idx

            range_start = data[int(start_idx)][0]
            range_end = data[int(end_idx)][0]
            duration = range_end - range_start
            rate = samples / (duration / 10 ** 6)

        result['channel'].append(source)
        result['name'].append(source.name)
        result['type'].append(source.units[0])
        result['units'].append(source.units[1])
        result['start'].append(range_start)
        result['end'].append(range_end)
        result['duration'].append(duration)
        result['samples'].append(samples)
        result['rate'].append(rate)

        # # TODO: RESTORE AFTER FIX IN idelib
        # dmin, dmean, dmax = data.getRangeMinMeanMax(start, end)
        # result['min'].append(dmin)
        # result['mean'].append(dmean)
        # result['max'].append(dmax)

    if formatting is False:
        return pd.DataFrame(result).style

    style = TABLE_FORMAT.copy()
    if timestamps:
        style.update({
            'start': format_timestamp,
            'end': format_timestamp
        })
    if isinstance(formatting, dict):
        style.update(formatting)

    styled = pd.DataFrame(result).style.format(style, precision=precision)
    if not index:
        return styled.hide_index()
    else:
        return styled

# ============================================================================
#
# ============================================================================

def get_utc_offset(dataset: idelib.dataset.Dataset) -> float:
    """
    Get a recorder's configured UTC time zone offset from an `idelib.Dataset`
    (i.e., an imported IDE file). Note that this is a user-configured option,
    and will be zero if the recorder did not have its UTC offset explicitly
    set.

    :param dataset: The IDE data from which to get the UTC offset.
    :return: The UTC offset, in seconds.
    """

    def crawl(parent):
        for el in parent:
            if el.name == 'ChannelDataBlock':
                return 0
            elif el.name in ('RecorderConfiguration', 'SSXBasicRecorderConfiguration'):
                return crawl(el)
            elif el.name == 'RecorderConfigurationList':
                for item in el:
                    data = item.dump()
                    if data.get('ConfigID') ==  0xBFF7F:
                        return data.get('IntValue', 0)
            elif el.name == 'UTCOffset':
                return el.value

        return 0

    return crawl(dataset.ebmldoc)


def to_pandas(
    channel: typing.Union[idelib.dataset.Channel, idelib.dataset.SubChannel],
    time_mode: typing.Literal["seconds", "timedelta", "datetime"] = "datetime",
    tz: typing.Union[pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo,
                     typing.Literal["device", "local", "utc"]] = "utc"
) -> pd.DataFrame:
    """ Read IDE data into a pandas DataFrame.

        :param channel: a `Channel` object, as produced from `Dataset.channels`
            or :py:func:`endaq.ide.get_channels`
        :param time_mode: how to temporally index samples; each mode uses either
            relative times (with respect to the start of the recording) or
            absolute times (i.e., date-times):

            * `"seconds"` - a `pandas.Float64Index` of relative timestamps, in seconds
            * `"timedelta"` - a `pandas.TimeDeltaIndex` of relative timestamps
            * `"datetime"` - a `pandas.DateTimeIndex` of absolute timestamps

        :param tz: Optional time zone information for displaying the `"datetime"` time
            mode. It can be a standard time zone object (`pytz.timezone`,
            `dateutil.tz.tzfile`, `datetime.tzinfo`) or one of three special strings:

            * `"utc"` - standard UTC time (default).
            * `"local"` - the  current computer's local time zone (note: may not be the
                user's actual time zone when used on enDAQ Cloud).
            * `"device"` - the time zone specified by the original recording device's
                configured UTC offset.

        :return: a `pandas.DataFrame` containing the channel's data
    """
    time_mode = str(time_mode).casefold()
    if time_mode not in ('seconds', 'timedelta', 'datetime'):
        raise ValueError(f'invalid time mode {time_mode!r}')

    data = channel.getSession().arraySlice()
    t, data = data[0], data[1:].T
    t = (1e3*t).astype("timedelta64[ns]")

    if time_mode == "datetime":
        index = pd.to_datetime(t + np.datetime64(channel.dataset.lastUtcTime, "s"), utc=True)
        index.name = "timestamp"

        tz = tz.casefold() if isinstance(tz, str) else tz
        if tz != "utc":
            if tz == "device":
                tz = dateutil.tz.tzoffset('device', get_utc_offset(channel.dataset))
            elif tz == "local":
                tz = datetime.datetime.now().astimezone().tzinfo

            index = index.tz_convert(tz)
            
    else:
        if time_mode == "seconds":
            t = t / np.timedelta64(1, "s")

        index = pd.Series(t, name="timestamp")

    if hasattr(channel, "subchannels"):
        columns = [sch.name for sch in channel.subchannels]
    else:
        columns = [channel.name]

    return pd.DataFrame(data, index=index, columns=columns)

# ============================================================================
#
# ============================================================================


def get_primary_sensor_data(  
    name: str = "",
    doc: idelib.dataset.Dataset = None,
    measurement_type: typing.Union[str, MeasurementType] = ANY,
    criteria: typing.Literal["samples", "rate", "duration"] = "samples",
    time_mode: typing.Literal["seconds", "timedelta", "datetime"] = "datetime",
    tz: typing.Union[pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo,
                     typing.Literal["device", "local", "utc"]] = "utc",
    least: bool = False,
    force_data_return: bool = False
) -> pd.DataFrame:
    """ Get the data from the primary sensor in a given .ide file using :py:func:`~endaq.ide.to_pandas()` 

        :param name: The file location to pull the data from, see :py:func:`~endaq.ide.get_doc()` 
            for more. This can be a local file location or a URL.
        :param doc: An open `Dataset` object, see :py:func:`~endaq.ide.get_doc()` 
            for more. If one is provided it will not attempt to use `name` to 
            load a new one.
        :param measurement_type: The sensor type to return data from, see :py:mod:`~endaq.ide.measurement`
            for more. The default is `"any"`, but to get the primary accelerometer
            for example, set this to `"accel"`.
        :param criteria: How to determine the "primary" sensor using the summary
            information provided by :py:func:`~endaq.ide.get_channel_table()`: 
        
            * `"sample"` - the number of samples, default behavior
            * `"rate"` - the sampling rate in Hz
            * `"duration"` - the duration from start to the end of data from that sensor
        :param time_mode: how to temporally index samples; each mode uses either
            relative times (with respect to the start of the recording) or
            absolute times (i.e., date-times):

            * `"seconds"` - a `pandas.Float64Index` of relative timestamps, in seconds
            * `"timedelta"` - a `pandas.TimeDeltaIndex` of relative timestamps
            * `"datetime"` - a `pandas.DateTimeIndex` of absolute timestamps
        :param tz: Optional time zone information for displaying the `"datetime"` time
            mode. It can be a standard time zone object (`pytz.timezone`,
            `dateutil.tz.tzfile`, `datetime.tzinfo`) or one of three special strings:

            * `"utc"` - standard UTC time (default).
            * `"local"` - the  current computer's local time zone (note: may not be the
                user's actual time zone when used on enDAQ Cloud).
            * `"device"` - the time zone specified by the original recording device's
                configured UTC offset.
        :param least: If set to `True` it will return the channels ranked lowest by
            the given criteria.
        :param force_data_return: If set to `True` and the specified `measurement_type`
            is not included in the file, it will return the data from any sensor 
            instead of raising an error which is the default behavior.

        :return: a `pandas.DataFrame` containing the sensor's data

        Here are some examples:
        
        .. code:: python3

            #Get sensor with the most samples, may return data of mixed units
            data = get_primary_sensor_data('https://info.endaq.com/hubfs/data/All-Channels.ide')

            #Instead get just the primary accelerometer data defined by number of samples
            accel = get_primary_sensor_data('https://info.endaq.com/hubfs/data/All-Channels.ide', measurement_type='accel')
    """
    
    #Get the doc object if it isn't provided
    if doc is None:
        doc = get_doc(name)
        
    #Get Channels of the Measurement Type
    channels = get_channel_table(doc, measurement_type).data
    
    #Raise error if measurement type isn't in the file
    if len(channels) == 0:
        error_str = f'measurement type "{measurement_type!r}" is not included in this file'
        if force_data_return:
            warnings.warn(error_str)
            channels = get_channel_table(doc, "any").data
        else:
            raise ValueError(error_str)
    
    #Filter based on criteria
    criteria = str(criteria).lower()
    if (criteria in ["samples", "rate", "duration"]):
        if least:
            channels = channels[channels[criteria] == channels[criteria].min()]
        else:
            channels = channels[channels[criteria] == channels[criteria].max()]
    else:        
        raise ValueError(f'invalid criteria "{criteria!r}"')
    
    #Get parent channel
    parent = channels.iloc[0].channel.parent
    
    #Get parent channel data
    data = to_pandas(parent, time_mode=time_mode, tz=tz)
    
    #Return only the subchannels with right units
    return data[channels.name]    
