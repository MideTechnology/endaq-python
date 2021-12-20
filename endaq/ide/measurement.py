"""
The module :py:mod:`endaq.ide.measurement` provides an abstract representation of measurement types for
easily retrieving specific data from ``.ide`` files. Several functions in :py:mod:`endaq.ide` accept
combinations of ``MeasurementType`` constants for filtering datasets by sensor type.

Measurement types are represented by a set of singleton instances of :py:class:`MeasurementType`.

.. code:: python3

    import endaq.ide
    from endaq.ide.measurement import *

    doc = endaq.ide.get_doc("https://info.endaq.com/hubfs/data/surgical-instrument.ide")
    endaq.ide.get_channels(doc, ACCELERATION)
    endaq.ide.get_channels(doc, TEMPERATURE+PRESSURE)

Strings may also be used, either in combination with or instead of, the :py:class:`MeasurementType` instances.
The strings can be abbreviated, but no shorter than three characters.

.. code:: python3

    endaq.ide.get_channels(doc, "acceleration")
    endaq.ide.get_channels(doc, "accel")
    endaq.ide.get_channels(doc, ACCELERATION+"temp")

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
    """ Singleton/sentinel marker object for filtering channels by measurement
        type.
    """
    # TODO: Fix nomenclature. "Singleton" may not be the correct term; there
    #  are multiple instances of MeasurementType, but not multiple *duplicate*
    #  ones. This may be closer to the "Sentinel" pattern, but the class has
    #  more functionality than just marking.

    types = {}  # Maps all measurement type 'key' strings to objects.
    names = {}  # Maps display names to objects.
    verbose = False

    def __new__(cls, *keys, labels=None, doc=None):
        """ Create and return a new object, if one with a matching `key` does
            not already exist. If it does, the previous instance is returned.
            Called prior to `__init__()`.

            The first argument is taken as the display name of the
            measurement type. Its first 3 characters are used as the default
            key (the string actually used in filtering). Functions that use
            `MeasurementType` objects can also use those key strings.
            Additional non-keyword arguments are taken as alternate keys.

            :param labels: A list/tuple of substrings to match against
                `Channel` and `SubChannel` unit labels, used to identify
                the appropriate `NeasurementType` instance. The `name` is
                automatically included.
            :param doc: A docstring for the `MeasurementType` instance.
            :returns: `MeasurementType`
        """
        obj = None
        name = keys[0].strip()
        for key in keys:
            key = str(key).strip().lower()[:3]
            obj = cls.types.get(key, None)
            if not obj:
                if name not in cls.names:
                    obj = super().__new__(cls)
                    obj._name = name
                    obj._key = key
                    obj._keys = set()  # Alternative keys for this object
                    if labels is not None:
                        labels = (labels.strip().lower(),) if isinstance(labels, str) else labels
                        labels = tuple(labels) if labels else ()
                        if name not in labels:
                            labels += (name.lower(),)
                    obj._labels = labels
                    obj.__doc__ = doc or name
                    cls.types[key] = obj
                    cls.names[name] = obj
                else:
                    obj = cls.names[name]
                    cls.types[key] = obj
            obj._keys.add(key)
        return obj

    def __str__(self):
        return self._key

    def __repr__(self):
        if not self.verbose:
            return f"<MeasurementType: {self._name}>"
        keys = ', '.join([repr(k) for k in self._keys if k != self.key])
        if keys:
            return f"<MeasurementType: {self._name} {self.key!r} (alt: {keys})>"
        return f"<MeasurementType: {self._name} {self.key!r})>"

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        # For interoperability with 'key' strings (as dict keys, etc.)
        # e.g., `mtype == "acc"`
        return hash(self._key)

    def __add__(self, other):
        # Concatenates as strings, so query-like sequences can be built.
        return f"{self} {other}"

    def __radd__(self, other):
        # Concatenates as strings, so query-like sequences can be built.
        return f"{other} {self}"

    def __sub__(self, other):
        # Concatenates as strings (with this one's negated)
        return f"{self} -{other}"

    def __rsub__(self, other):
        # Concatenates as strings (with this one's negated)
        return f"{other} -{self}"

    def __or__(self, other):
        # Same as __add__(), but convenient for those used to using bitwise OR to combine flags
        return self + other

    def __neg__(self):
        # Negates (appends a `-`) so query-like strings can be built.
        return f"-{self.key}"

    def __getitem__(self, *args, **kwargs):
        # For basic required string interoperability
        return self._key.__getitem__(*args, **kwargs)

    def upper(self):
        # For basic required string interoperability
        return self._key.upper()

    def lower(self):
        # For basic required string interoperability
        return self._key.lower()

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
            raise TypeError(f"Cannot compare measurement types with {channel} ({type(channel)})")

        mt = mt.lower()
        return any(label in mt or fnmatch(mt, label) for label in self._labels)

# ============================================================================
#
# ============================================================================


ANY = MeasurementType("Any/all", "*", "any", "all",
    doc="Marker object for matching any/all measurement types",
    labels=("*",))
""" Marker object for matching any/all measurement types """

ACCELERATION = MeasurementType("Acceleration", "acc", "g",
    doc="Marker object for filtering channels with acceleration data",
    labels=())
""" Marker object for filtering channels with acceleration data """

ORIENTATION = MeasurementType("Orientation", "imu", "qua", "gyr",
    doc="Marker object for filtering channels with rotation/orientation data",
    labels=("quaternion", "euler", "orientation"))
"Marker object for filtering channels with rotation/orientation data"

ROTATION = MeasurementType("Rotation", "rot", "ang",
    doc="Marker object for filtering channels with angular change rate data",
    labels=("rotation", "gyro"))
"Marker object for filtering channels with angular change rate data"

AUDIO = MeasurementType("Audio", "mic",
    doc="Marker object for filtering channels with sound level data",
    labels=("mic",))
"Marker object for filtering channels with sound level data"

LIGHT = MeasurementType("Light", "lux",
    doc="Marker object for filtering channels with light intensity data",
    labels=("lux", "uv"))
"Marker object for filtering channels with light intensity data"

PRESSURE = MeasurementType("Pressure",
    doc="Marker object for filtering channels with air pressure data",
    labels=())  # pressures
"Marker object for filtering channels with air pressure data"

TEMPERATURE = MeasurementType("Temperature",
    doc="Marker object for filtering channels with temperature data",
    labels=())  # temperature
"Marker object for filtering channels with temperature data"

HUMIDITY = MeasurementType("Relative Humidity", "hum",
    doc="Marker object for filtering channels with (relative) humidity data",
    labels=())  # Humidity
"Marker object for filtering channels with (relative) humidity data"

LOCATION = MeasurementType("Location", "pos", "gps",
    doc="Marker object for filtering channels with location data",
    labels=("pos",))  # GPS
"Marker object for filtering channels with location data"

SPEED = MeasurementType("Speed",
    doc="Marker object for filtering channels with rate-of-speed data",
    labels=("velocity",))  # GPS Ground Speed
"Marker object for filtering channels with rate-of-speed data"

TIME = MeasurementType("Time", "epo",
    doc="Marker object for filtering channels with time data",
    labels=("epoch",))  # GPS Epoch Time
"Marker object for filtering channels with time data"

# For potential future use
GENERIC = MeasurementType("Generic/Unspecified", "adc", "raw",
    labels=("adc", "raw"))
ALTITUDE = MeasurementType("Altitude",
    doc="Marker object for filtering channels with altitude data",
    labels=())
VOLTAGE = MeasurementType("Voltage",
    doc="Marker object for filtering channels with voltmeter data",
    labels=("volt",))
DIRECTION = MeasurementType("Direction",
    doc="Marker object for filtering channels with 2D directional data",
    labels=("compass", "heading"))
MAGNETIC = MeasurementType("Magnetic Field", "emf",
    doc="Marker object for filtering channels with magnetic field strength data",
    labels=("emf", "magnetic"))
FREQUENCY = MeasurementType("Frequency",
    doc="Marker object for filtering channels with frequency data",
    labels=("rate",))

# Synonyms, for convenience.
# TODO: Include abbreviations (e.g., TEMP = TEMPERATURE) for convenience? Current names are long.
ADC = GENERIC
ANG_RATE = ROTATION
GYRO = ROTATION
RAW = GENERIC
QUATERNION = ORIENTATION


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
        :returns: A pair of sets of `MeasurementType` instances: ones to
            include, and ones to exclude.
    """
    query = str(query).lower().strip()
    if query == "*":
        return set(MeasurementType.types.values()), set()

    inc = set()
    exc = set()
    prev = None

    # NOTE: Casting to string and parsing isn't always required, but this
    #   isn't used often enough to require high performance optimization.
    for token in shlex(query):
        token = token.lower()[:3]
        if token in MeasurementType.types:
            if prev == "-":
                exc.add(MeasurementType.types[token])
            else:
                inc.add(MeasurementType.types[token])
        elif token not in "+-":
            raise TypeError(f"Unknown measurement type: {token!r}")
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
        channels = channels.values()

    if measurement_type == ANY:
        return list(channels)

    # Note: if no `inc`, only `exc` is used
    inc, exc = split_types(measurement_type)

    result = []
    for ch in channels:
        thisType = get_measurement_type(ch)

        if isinstance(thisType, (list, set, tuple)):
            # `ch` is a Channel with SubChannels
            if exc.intersection(thisType):
                # One or more subchannels excluded; exclude channel.
                continue
            elif not inc or inc.intersection(thisType):
                result.append(ch)
                continue

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
