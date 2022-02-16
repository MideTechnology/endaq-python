"""
Some general-purpose IDE file manipulation funcions.
"""
import datetime
import string
from ebmlite import loadSchema

__all__ = ['parse_time', 'validate']


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


