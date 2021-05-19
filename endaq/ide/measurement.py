"""
Functions for filtering data by measurement types, and singleton objects
representing different measurement types.
"""

__all__ = ['ANY', 'ACCELERATION', 'ALTITUDE', 'ANG_RATE', 'AUDIO', 'DIRECTION',
           'FREQUENCY', 'GENERIC', 'GYRO', 'HUMIDITY', 'LIGHT', 'LOCATION',
           'MAGNETIC', 'ORIENTATION', 'PRESSURE', 'ROTATION', 'SPEED',
           'TEMPERATURE', 'TIME', 'VOLTAGE',
           'get_measurement_type', 'filter_channels', 'get_channels']

from fnmatch import fnmatch
from shlex import shlex

from idelib.dataset import Dataset, Channel, SubChannel

# ============================================================================
#
# ============================================================================


class MeasurementType:
    """ Singleton marker object for filtering channels by measurement type.
    """
    # TODO: Fix nomenclature. "Singleton" may not be the correct term; there are multiple instances
    #   of MeasurementType, but not multiple *duplicate* ones.

    types = {}
    verbose = False

    def __new__(cls, name, key, labels=None, doc=None):
        """ Create and return a new object, if one with a matching `key` does
            not already exist. If it does, the previous instance is returned.
            Called prior to `__init__()`.

            :param name: The display name of the measurement type.
            :param key: The abbreviated 'key' string for the type. Functions
                that use `MeasurementType` objects can also use those key
                strings.
            :param labels: A list/tuple of substrings to match against
                `Channel` and `SubChannel` unit labels, used to identify
                the appropriate `NeasurementType` instance. The `name` is
                automatically included.
            :param doc: A docstring for the `MeasurementType` instance.
            :returns: `MeasurementType`
        """
        key = str(key).lower()
        if key not in cls.types:
            obj = super().__new__(cls)
            obj._name = name
            obj._key = key
            if labels is not None:
                labels = (labels.lower(),) if isinstance(labels, str) else labels
                labels = tuple(labels) if labels else ()
                if name not in labels:
                    labels += (name.lower(),)
            obj._labels = labels
            obj.__doc__ = doc or name
            cls.types[key] = obj
        return cls.types[key]

    def __str__(self):
        return self._key

    def __repr__(self):
        if self.verbose:
            return "<Measurement Type: %s (%r)>" % (self._name, self._key)
        return "<Measurement Type: %s>" % self._name

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        # For interoperability with 'key' strings (as dict keys, etc.)
        # e.g., `mtype == "acc"`
        return hash(self._key)

    def __add__(self, other):
        # Concatenates as strings, so query-like sequences can be built.
        return "%s %s" % (self, other)

    def __radd__(self, other):
        # Concatenates as strings, so query-like sequences can be built.
        return "%s %s" % (other, self)

    def __sub__(self, other):
        # Concatenates as strings (with this one's negated)
        return "%s -%s" % (self, other)

    def __rsub__(self, other):
        # Concatenates as strings (with this one's negated)
        return "%s -%s" % (other, self)

    def __or__(self, other):
        # Same as __add__(), but convenient for those used to using bitwise OR to combine flags
        return self + other

    def __neg__(self):
        # Negates (appends a `-`) so query-like strings can be built.
        return "-%s" % self.key

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

    def match(self, channel):
        """ Does the given object match this measurement type?

            :param channel: The object to test. Can be an `idelib.dataset.Channel`,
                `idelib.dataset.SubChannel`, or a or a string of a measurement
                type name.
            :return: `True` if the given object uses this measurement type.
                In the case of `idelib.dataset.Channel`, `True` is returned
                if any of its subchannels match.
        """
        if getattr(channel, '_measurementType', channel) == self:
            return True

        if isinstance(channel, Channel):
            if not isinstance(channel, SubChannel):
                return any(self.match(c) for c in channel.children)
            else:
                mt = channel.units[0]
        elif isinstance(channel, str):
            mt = channel
        else:
            raise TypeError("Cannot compare measurement types with %r (%s)" %
                            (channel, type(channel)))

        mt = mt.lower()
        return any(label in mt or fnmatch(mt, label) for label in self._labels)

# ============================================================================
#
# ============================================================================


ANY = MeasurementType("Any/all", "*",
    doc="Marker object for matching any/all measurement types",
    labels=("*",))

ACCELERATION = MeasurementType("Acceleration", "acc",
    doc="Marker object for filtering channels with acceleration data",
    labels=())
ORIENTATION = MeasurementType("Orientation", "rot",
    doc="Marker object for filtering channels with rotation/orientation data",
    labels=("rotation", "quaternion", "euler", "orientation"))
AUDIO = MeasurementType("Audio", "mic",
    doc="Marker object for filtering channels with sound level data",
    labels=("mic",))
LIGHT = MeasurementType("Light", "lux",
    doc="Marker object for filtering channels with light intensity data",
    labels=("lux", "uv"))
PRESSURE = MeasurementType("Pressure", "pre",
    doc="Marker object for filtering channels with air pressure data",
    labels=())  # pressures
TEMPERATURE = MeasurementType("Temperature", "tmp",
    doc="Marker object for filtering channels with temperature data",
    labels=())  # temperature
HUMIDITY = MeasurementType("Humidity", "hum",
    doc="Marker object for filtering channels with (relative) humidity data",
    labels=())  # Humidity
LOCATION = MeasurementType("Location", "gps",
    doc="Marker object for filtering channels with location data",
    labels=("pos",))  # GPS
SPEED = MeasurementType("Speed", "spd",
    doc="Marker object for filtering channels with rate-of-speed data",
    labels=("velocity",))  # GPS Ground Speed
TIME = MeasurementType("Time", "epo",
    doc="Marker object for filtering channels with time data",
    labels=("epoch",))  # GPS Epoch Time

# For potential future use
GENERIC = MeasurementType("Generic/Unspecified", "???",
    labels=("adc",))
ANG_RATE = MeasurementType("Angular Rate", "ang",
    doc="Marker object for filtering channels with angular change rate data",
    labels=("gyro"))
ALTITUDE = MeasurementType("Altitude", "alt",
    doc="Marker object for filtering channels with altitude data",
    labels=())
VOLTAGE = MeasurementType("Voltage", "vol",
    doc="Marker object for filtering channels with voltmeter data",
    labels=("volt",))
DIRECTION = MeasurementType("Direction", "dir",
    doc="Marker object for filtering channels with 2D directional data",
    labels=("compass", "heading"))
MAGNETIC = MeasurementType("Magnetic Field", "emf",
    doc="Marker object for filtering channels with magnetic field strength data",
    labels=("emf", "magnetic"))
FREQUENCY = MeasurementType("Frequency", "fre",
    doc="Marker object for filtering channels with frequency data",
    labels=("rate",))

# Synonyms, for convenience.
# TODO: Include abbreviations (e.g., TEMP = TEMPERATURE) for convenience? Current names are long.
ROTATION = ORIENTATION
GYRO = ANG_RATE


# ============================================================================
#
# ============================================================================

def get_measurement_type(channel):
    """ Get the appropriate `MeasurementType` object for a given `SubChannel`.
        Calling with a `Channel` returns a list of `MeasurementType` objects,
        with one for each child `SubChannel`.

        :param channel: A `Channel` or `SubChannel` instance (e.g., from a
            `Dataset`).
        :returns: A `MeasurementType` object (for a `SubChannel`), or a list
            of `MeasurentType` objects (one for each child) if a `Channel`
            was supplied.
    """
    mt = getattr(channel, "_measurementType", None)
    if mt:
        return mt

    if channel.children:
        # Note: this caches the MeasurementType in the Channel/SubChannel; the
        # attribute `_measurementType` must be settable
        channel._measurementType = [get_measurement_type(c) for c in channel.children]
        return channel._measurementType
    else:
        for m in MeasurementType.types.values():
            if m == ANY:
                continue
            if m.match(channel):
                channel._measurementType = m
                return m

    return None


def split_types(query):
    """ Split a string of multiple `MeasurementType` keys (e.g., one generated
        by using addition or subtraction of `MeasurementType` objects and/or
        strings).

        :param query: A `MeasurementType` or a string containing multiple
            `MeasurementType` keys. A key can be excluded by prefixing it with
            a ``-``.
        :returns: A pair of lists of `MeasurementType` instances: ones to
            include, and ones to exclude.
    """
    query = str(query).lower().strip()
    if query == "*":
        return list(MeasurementType.types.values()), []

    inc = []
    exc = []
    prev = None

    # NOTE: Casting to string and parsing isn't always required, but this
    #   isn't used often enough to require high performance optimization.
    for token in shlex(query):
        if token in MeasurementType.types:
            if prev == "-":
                exc.append(MeasurementType.types[token])
            else:
                inc.append(MeasurementType.types[token])
        elif token not in "+-":
            raise TypeError("Unknown measurement type: %r" % token)
        prev = token
    return inc, exc


def filter_channels(channels, measurement_type=ANY):
    """ Filter a list of `Channel` and/or `SubChannel` instances by their
        measurement type(s).

        :param channels: A list or dictionary of channels/subchannels to filter.
        :param measurement_type: A `MeasurementType`, a measurement type 'key'
            string, or a string of multiple keys generated by adding and/or
            subtracting `MeasurementType` objects. Any 'subtracted' types
            will be excluded.
    """
    # Note: This is separated from `get_channels()` so it can be used
    # elsewhere (i.e., to retrieve channels by type from a `Recorder`)
    if isinstance(channels, dict):
        channels = list(channels.values())

    if measurement_type == ANY:
        return channels[:]

    # Note: if no `inc`, only `exc` is used
    inc, exc = split_types(measurement_type)

    result = []
    for ch in channels:
        thisType = get_measurement_type(ch)
        if isinstance(thisType, (list, tuple)):
            for t in thisType:
                if t not in exc:
                    if not inc or t in inc:
                        result.append(ch)
                        break
        elif thisType not in exc:
            if not inc or thisType in inc:
                result.append(ch)

    return result


def get_channels(dataset, measurement_type=ANY, subchannels=True):
    """ Get a list of `Channel` or `SubChannel` instances from a `Dataset` by
        their measurement type(s).

        :param dataset: The `Dataset` from which to retrieve the list.
        :param measurement_type: A `MeasurementType`, a measurement type 'key'
            string, or a string of multiple keys generated by adding and/or
            subtracting `MeasurementType` objects. Any 'subtracted' types
            will be excluded.
        :param subchannels: If `False`, get only `Channel` objects. If `True`,
            get only `SubChannel` objects.
        :returns: A list of matching `SubChannel` instances from the `Dataset`.
    """
    # This is really simple; it is a convenience for users, particularly
    # novices for whom the use of `getPlots()` may not be obvious.
    # TODO: Should there only be `filter_channels()`, and have it do `if isinstance(dataset, Dataset):`?
    #   The `subchannels` argument isn't applicable to non-Datasets, however, which would be weird.
    if subchannels:
        return filter_channels(dataset.getPlots(sort=False), measurement_type)
    return filter_channels(dataset.channels, measurement_type)
