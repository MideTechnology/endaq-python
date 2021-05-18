"""
Functions for filtering data by measurement types, and singleton objects
representing different measurement types.
"""

__all__ = ['ANY', 'ACCELERATION', 'ALTITUDE', 'ANG_RATE', 'AUDIO', 'DIRECTION',
           'GENERIC', 'HUMIDITY', 'LIGHT', 'LOCATION', 'MAGNETIC',
           'ORIENTATION', 'PRESSURE', 'ROTATION', 'SPEED', 'TEMPERATURE',
           'TIME', 'VOLTAGE', 'get_measurement_type', 'filter_channels',
           'get_channels']

from shlex import shlex

# ============================================================================
#
# ============================================================================

class MeasurementType:
    """ Singleton marker object for getting channels by measurement type.
    """
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
            labels = (labels.lower(),) if isinstance(labels, str) else labels
            obj._labels = tuple(labels) if labels else ()
            obj.__doc__ = doc or name
            if name not in obj._labels:
                obj._labels += (name,)
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

    def __neg__(self):
        # Negates (appends a `-`) so query-like strings can be built.
        return "-%s" % self.key

    @property
    def name(self):
        return self._name

    @property
    def key(self):
        return self._key

# ============================================================================
#
# ============================================================================

""" does this do anything in Python? """
ANY = MeasurementType("Any/all", "*",
    doc="Marker object for matching any/all measurement types")

ACCELERATION = MeasurementType("Acceleration", "acc",
    doc="Marker object for getting channels of with acceleration data",
    labels=())
ORIENTATION = MeasurementType("Orientation", "rot",
    doc="Marker object for getting channels of with rotation/orientation data",
    labels=("rotation", "quaternion", "euler", "orientation"))
AUDIO = MeasurementType("Audio", "mic",
    doc="Marker object for getting channels of with sound level data",
    labels=("mic",))
LIGHT = MeasurementType("Light", "lux",
    doc="Marker object for getting channels of with light intensity data",
    labels=("lux", "uv"))
PRESSURE = MeasurementType("Pressure", "pre",
    doc="Marker object for getting channels of with air pressure data",
    labels=())  # pressures
TEMPERATURE = MeasurementType("Temperature", "tmp",
    doc="Marker object for getting channels of with temperature data",
    labels=())  # temperature
HUMIDITY = MeasurementType("Humidity", "hum",
    doc="Marker object for getting channels of with (relative) humidity data",
    labels=())  # Humidity
LOCATION = MeasurementType("Location", "gps",
    doc="Marker object for getting channels of with location data",
    labels=())  # GPS
SPEED = MeasurementType("Speed", "spd",
    doc="Marker object for getting channels of with rate-of-speed data",
    labels=("velocity",))  # GPS Ground Speed
TIME = MeasurementType("Time", "epo",
    doc="Marker object for getting channels of with time data",
    labels=("epoch",))  # GPS Epoch Time

# Synonyms
ROTATION = ORIENTATION

# For potential future use
GENERIC = MeasurementType("Generic/Unspecified", "???",
    labels=("adc",))
ANG_RATE = MeasurementType("Angular Rate", "ang",
    doc="Marker object for getting channels of with angular change rate data",
    labels=("gyro"))
ALTITUDE = MeasurementType("Altitude", "alt",
    doc="Marker object for getting channels of with altitude data",
    labels=())
VOLTAGE = MeasurementType("Voltage", "vol",
    doc="Marker object for getting channels of with voltmeter data",
    labels=("volt",))
DIRECTION = MeasurementType("Direction", "dir",
    doc="Marker object for getting channels of with 2D directional data",
    labels=("compass", "heading"))
MAGNETIC = MeasurementType("Magnetic Field", "emf",
    doc="Marker object for getting channels of with magnetic field strength data",
    labels=("emf", "magnetic"))


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
    # Note: this caches the MeasurementType in the Channel/SubChannel; the
    # attribute `_measurementType` must be settable
    mt = getattr(channel, "_measurementType", None)
    if mt:
        return mt

    if channel.children:
        mt = [get_measurement_type(c) for c in channel.children]
    else:
        for m in MeasurementType.types:
            for label in m._labels:
                # Check MeasurementType's label substrings for match. Could
                # also/alternately use `fnmatch` for more complex patterns.
                if label in channel.units[0].lower():
                    mt = m
                    break
    if mt:
        channel._measurementType = mt

    return mt


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

        :param channels: A list of channels/subchannels to filter.
        :param measurement_type: A `MeasurementType`, a measurement type 'key'
            string, or a string of multiple keys generated by adding and/or
            subtracting `MeasurementType` objects. Any 'subtracted' types
            will be excluded.
    """
    # Note: This is separated from `get_channels()` so it can be used
    # elsewhere (i.e., to retrieve channels by type from a `Recorder`)
    if measurement_type == ANY:
        return channels[:]

    inc, exc = split_types(measurement_type)
    if not inc:
        # Only exclusions (if any); include everything.
        inc = list(MeasurementType.types.values())

    result = []
    for ch in channels:
        thisType = get_measurement_type(ch)
        if isinstance(thisType, (list, tuple)):
            for t in thisType:
                if t not in exc and t in inc:
                    result.append(ch)
                    break
        elif thisType not in exc and thisType in inc:
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
    if subchannels:
        return filter_channels(dataset.getPlots(sort=False), measurement_type)
    return filter_channels(list(dataset.channels.values()), measurement_type)
