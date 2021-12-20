"""
Functions for retrieving summary data from a dataset.
"""
from __future__ import annotations
import typing

from collections import defaultdict
import datetime
import string
import warnings

import numpy as np
import pandas as pd
import idelib

from .measurement import ANY, get_channels


__all__ = [
    "get_channel_table",
    "to_pandas",
]


# ============================================================================
# Display formatting functions
# ============================================================================

def parse_time(t, datetime_start=None):
    """ Convert a time in one of several user-friendly forms to microseconds
        (the native time units used in `idelib`). Valid types are:

        * `None`, `int`, or `float` (returns the same value)
        * `str` (formatted as a time, e.g., `MM:SS`, `HH:MM:SS`,
          `DDd HH:MM:SS`). More examples:

          * ``":01"`` or ``":1"`` or ``"1s"`` (1 second)
          * ``"22:11"`` (22 minutes, 11 seconds)
          * ``"3:22:11"`` (3 hours, 22 minutes, 11 seconds)
          * ``"1d 3:22:11"`` (3 hours, 22 minutes, 11 seconds)
        * `datetime.timedelta` or `pandas.Timedelta`
        * `datetime.datetime`

        :param t: The time value to convert.
        :param datetime_start: If `t` is a `datetime` object, the result will
            be relative to `datetime_start`. It will default to the start of
            the day portion of `t`. This has no effect on non-`datetime`
            values of `t` .
        :returns: The time in microseconds.
    """
    # TODO: Put this somewhere else? It will be useful elsewhere, and shouldn't
    #   be bound to the `pandas` requirement in this module.

    if t is None or isinstance(t, (int, float)):
        return t

    elif isinstance(t, str):
        if not t:
            return None
        orig = t
        t = t.strip().lower()
        for c in ":dhms":
            t = t.replace(c, ' ')
        if not all(c in string.digits + ' ' for c in t):
            raise ValueError(f"Bad time string for parse_time(): {orig!r}")

        micros = 0
        for part, mult in zip(reversed(t.split()), (1, 60, 3600, 86400)):
            if not part:
                continue
            part = part.strip(string.ascii_letters + string.punctuation + string.whitespace)
            micros += float(part) * mult
        return micros * 10**6

    elif isinstance(t, datetime.timedelta):
        return t.total_seconds() * 10**6

    elif isinstance(t, (datetime.time, datetime.datetime)):
        if datetime_start is None:
            # No starting time, assume midnight of same day.
            datetime_start = datetime.datetime(t.year, t.month, t.day)

        if isinstance(t, datetime.time):
            # just time: make datetime
            t = datetime.datetime.combine(datetime_start, t)

        if isinstance(t, datetime.datetime):
            # datetime: make timedelta
            return (t - datetime_start).total_seconds() * 10**6

    raise TypeError(f"Unsupported type for parse_time(): {type(t).__name__} ({t!r})")


# ============================================================================
# Display formatting functions
# ============================================================================

def format_channel_id(ch):
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


def format_timedelta(val):
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


def format_timestamp(ts):
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


def get_channel_table(dataset, measurement_type=ANY, start=0, end=None,
                      formatting=None, index=True, precision=4,
                      timestamps=False, **kwargs):
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
        :param measurement_type: A `MeasurementType`, a measurement type
            'key' string, or a string of multiple keys generated by adding
            and/or subtracting `MeasurementType` objects to filter the
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


def to_pandas(
    channel: typing.Union[idelib.dataset.Channel, idelib.dataset.SubChannel],
    time_mode: typing.Literal["seconds", "timedelta", "datetime"] = "datetime",
) -> pd.DataFrame:
    """ Read IDE data into a pandas DataFrame.

        :param channel: a `Channel` object, as produced from `Dataset.channels`
            or `endaq.ide.get_channels`
        :param time_mode: how to temporally index samples; each mode uses either
            relative times (with respect to the start of the recording) or
            absolute times (i.e., date-times):

            * `"seconds"` - a `pandas.Float64Index` of relative timestamps, in seconds
            * `"timedelta"` - a `pandas.TimeDeltaIndex` of relative timestamps
            * `"datetime"` - a `pandas.DateTimeIndex` of absolute timestamps

        :return: a `pandas.DataFrame` containing the channel's data
    """
    data = channel.getSession().arraySlice()
    t, data = data[0], data[1:].T

    t = (1e3*t).astype("timedelta64[ns]")
    if time_mode == "seconds":
        t = t / np.timedelta64(1, "s")
    elif time_mode == "datetime":
        t = t + np.datetime64(channel.dataset.lastUtcTime, "s")
    elif time_mode != "timedelta":
        raise ValueError(f'invalid time mode "{time_mode}"')

    if hasattr(channel, "subchannels"):
        columns = [sch.name for sch in channel.subchannels]
    else:
        columns = [channel.name]

    return pd.DataFrame(data, index=pd.Series(t, name="timestamp"), columns=columns)
