"""
Some general-purpose IDE file manipulation funcions.
"""
import datetime
import string
from ebmlite import loadSchema
from typing import Union

from .measurement import ACCELERATION
import idelib.dataset
import numpy as np


__all__ = ['parse_time', 'validate', 'get_accelerometer_bounds', 'get_accelerometer_info']


# ============================================================================
#
# ============================================================================

def validate(stream, from_pos=False, lookahead=25, percent=.5):
    """
    Determine if a stream contains IDE data.

    :param stream: A file-like stream (something that supports the methods
        `tell()` and `seek()`).
    :param from_pos: If `True`, validation of the stream will start at its
        current position. If `False` (default), the validation will start
        from its beginning.
    :param lookahead: The number of EBML elements to check.
    :param percent: The minumum percentage of EBML elements identified as
        being part of the IDE schema for validation. A small number of
        unknown elements may not indicate an invalid file; it may simply
        have been created using a different version of the schema.
    :return: `True` if validation passed, `False` if it failed.
    """
    # TODO: Make validation more thorough, test for corrupt files?

    orig_pos = stream.tell()
    if not from_pos:
        stream.seek(0)

    try:
        schema = loadSchema('mide_ide.xml')
        doc = schema.load(stream, headers=True)

        # Basic test: is it EBML data in the expected schema?
        known = 0
        for idx, el in enumerate(doc):
            if idx >= lookahead:
                break
            if el.id in schema.elements:
                known += 1
        if known < lookahead * percent:
            return False

        return True

    finally:
        stream.seek(orig_pos)


# ============================================================================
#
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
#
# ============================================================================

def get_accelerometer_bounds(ch: Union[idelib.dataset.Channel, idelib.dataset.SubChannel]) -> tuple[int, int]:
    """
    Gets the g-rating of the sensor used from the given channel.
    
    :param ch: The channel of the dataset to work on. This channel should have a child, which will
        be used to extract the data.

    :return: a tuple of g-rating bounds. 
    """
    ch = ch if isinstance(ch, idelib.dataset.SubChannel) else ch[0]
    if not ACCELERATION.match(ch):
        raise ValueError("An acceleration channel should be given")
    s_name = ch.sensor.name

    if s_name.startswith("ADXL"):
        g = {"ADXL345": 16, "ADXL362": 16,
             "ADXL357": 40, "ADXL359": 40,
             "ADXL355": 8, "ADXL375": 200,
             }[s_name.split(" ")[0]]
        return (-1 * g, g)
    
    values = np.asarray([8, 16, 25, 100, 200, 500, 2000, 6000])
    #finds the value closest to ch.transform(0, 65535)[1]. This is because 
    #we PR have some resistance and won't by default give the value we want,
    #and digital sensors do not return perfect values.
    g = values[np.abs(np.asarray(values) - ch.transform(0, 65535)[1]).argmin()]
    return (-1 * g, g)

# ============================================================================
#
# ============================================================================

def get_accelerometer_info(ch: idelib.dataset.Channel) -> dict:
    """
    creates a dictionary with information relevant to :py:func:`refine_acceleration`, namely
    
    - sensor_type: Literal["DC", "PE", "PR"]. Indicates if the sensor is digital, piezoelectric, 
        or piezoresistive respectively
    - rating: the g rating of the sensor, or the maximum it can read while accurate
    - noise: float. Indicates the noise of the sensor when the response curve is flat
    - low_cutoff: the lowest rate at which the sensor is being shaken at where the response curve is flat.
    - high_cutoff: the highest rate at which the sensor is being shaken at where the response curve is flat.

    :param ch: The channel of the dataset to work on. This channel should have a child, which will
        be used to extract the data. This sensor must 

    :return: a dictionary with the information stated above
    """
    if not ACCELERATION.match(ch):
        raise ValueError("An acceleration channel should be given")

    times = ch.getSession().arraySlice()[0, :]
    #same thing as 1/ ((times[1] - times[0]) / 1000000)
    sample_rate = 1000000 / (times[1] - times[0]) 
    sensor = ch[0].sensor.name
    kwords = sensor.split(" ") 
    if kwords[0].startswith("ADXL"):
        s_type = "DC"
    else:
        s_type = kwords[1]
    rating = get_accelerometer_bounds(ch)[1]

    if (s_type == "PE"):
        low = 10
        high = int(sample_rate / 5) 
        try:
            noise = {25: 8E-4, 100: 3E-3, 500: 1.5E-2, 2000: 0.06, 6000: 0.08}[rating]
        except KeyError:
            raise ValueError(f"rating {rating} not supported for Piezoelectric sensors")
    elif (s_type == "DC"):
            try:
                low, high = {8: (1, 150), 16: (1,300), 40: (1, 100)}[rating]
                noise = {8: 2E-5, 16: 4E-3, 40: 8E-5}[rating]
            except KeyError:
                raise ValueError(f"rating {rating} not supported for digital IMUs")
    elif (s_type == "PR"):
        low = 1
        high = int(sample_rate / 5) 
        try:
            noise = {50: 3E-3, 500: 1.5E-2, 2000:6E-2}[rating]
        except:
            raise ValueError(f"rating {rating} not supported for Piezoresistve sensors")
    else:
        raise ValueError(f"Sensor type {s_type} not recognized, should be one of PE, DC, PR.")
        
    return {
        "sensor_type": s_type,
        "noise": noise, 
        "rating": rating, 
        "low_cutoff": low, 
        "high_cutoff": high,
        }