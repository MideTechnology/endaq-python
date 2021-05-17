"""
"""

__all__ = ['ACCELERATION', 'ALTITUDE', 'ANG_RATE', 'AUDIO', 'DIRECTION',
           'GENERIC', 'HUMIDITY', 'LIGHT', 'LOCATION', 'MAGNETIC',
           'ORIENTATION', 'PRESSURE', 'ROTATION', 'SPEED', 'TEMPERATURE',
           'TIME', 'VOLTAGE', 'get_measurement_type', 'split_types']

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
        key = str(key).lower()
        if key not in cls.types:
            obj = super().__new__(cls)
            obj._name = name
            obj._key = key
            labels = (labels.lower(),) if isinstance(labels, str) else labels
            obj._labels = tuple(labels) or ()
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
    """ Get the appropriate `MeasurementType` instance for a given `SubChannel`.
        Calling with a `Channel` returns a list of `MeasurementType`.
    """
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
    """
    """
    inc = []
    exc = []
    prev = None
    for token in shlex(str(query).lower()):
        if token in MeasurementType.types:
            if prev == "-":
                exc.append(MeasurementType.types[token])
            else:
                inc.append(MeasurementType.types[token])
        elif token != "-":
            raise TypeError("Unknown measurement type: %r" % token)
        prev = token
    return inc, exc
