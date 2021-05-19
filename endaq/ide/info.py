"""
Functions for retrieving summary data from a dataset.
"""
from collections import defaultdict
from datetime import timedelta

from idelib.dataset import Channel
import pandas as pd

from .measurement import ANY, MeasurementType, get_channels


# ============================================================================
# Display formatting functions
# ============================================================================

def format_timedelta(td):
    """ Function for formatting the duration. Somewhat more condensed than
        the standard Pandas formatting.

        :param td: The `pandas.Timedelta` or `datetime.timedelta` to format.
        :return: A formatted time 'duration' string.
    """
    try:
        # NOTE: `components` only exists in pandas `Timedelta`, automatically
        #   converted from `datetime.timedelta` by Pandas.
        c = td.components
        s = "{c.minutes:02d}:{c.seconds:02d}.{c.milliseconds}".format(c=c)
        if c.hours or c.days:
            s = "{c.hours:02d}:{s}".format(c=c, s=s)
            if c.days:
                s = "{c.days}d {s}".format(c=c, s=s)
        return s
    except (AttributeError, TypeError, ValueError):
        return str(td)


def format_timestamp(ts):
    """ Function for formatting start/end timestamps. Somewhat more condensed
        than the standard Pandas formatting.

        :param ts: The timestamps in microseconds. Rendered as integers, since
            `idelib` timestamps have whole microsecond resolution.
        :return: A formatted timestamp string, with units.
    """
    return "%d µs" % ts

# ============================================================================
#
# ============================================================================


TABLE_STYLE = {
    'start': format_timestamp,
    'end': format_timestamp,
    'duration': format_timedelta,
    'rate': "{:.2f} Hz",
}


def get_channel_table(dataset, measurement_type=ANY, formatting=None,
                      index=True, **kwargs):
    """ Get summary data for all `SubChannel` objects in a `Dataset` that
        contain one or more type of sensor data.

        :param dataset: A `idelib.dataset.Dataset` or a list of
            channels/subchannels from which to build the table.
        :param measurement_type: A `MeasurementType`, a measurement type
            'key' string, or a string of multiple keys generated by adding
            and/or subtracting `MeasurementType` objects to filter the
            results. Any 'subtracted' types will be excluded.
        :param formatting: A dictionary of additional style/formatting
            items (see `pandas.DataFrame.style.format()`).
        :param index: If `True`, show the index column on the left.
        :returns: A table (`DataFrame`) of summary data.
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
        result['Channel ID'].append(source.parent.id)
        result['Subchannel ID'].append(source.id)
        result['name'].append(source.name)
        result['type'].append(source.units[0])
        result['units'].append(source.units[1])

        data = source.getSession(session)
        samples = len(data)
        if samples:
            start = data[0][0]
            end = data[-1][0]
            duration = timedelta(microseconds=end-start)
        else:
            start = end = duration = None
        result['start'].append(start)
        result['end'].append(end)
        result['duration'].append(duration)
        result['samples'].append(samples)
        result['rate'].append(samples / ((end - start) / 10**6))

    if formatting is False:
        return pd.DataFrame(result)

    if isinstance(formatting, dict):
        style = TABLE_STYLE.copy()
        style.update(formatting)
    else:
        style = TABLE_STYLE

    styled = pd.DataFrame(result).style.format(style)
    if not index:
        return styled.hide_index()
    else:
        return styled
