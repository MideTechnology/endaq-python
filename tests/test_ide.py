import os.path
import unittest
import warnings

from idelib.importer import importFile
from endaq.ide import info, measurement


class MeasurementTypeTests(unittest.TestCase):
    """ Basic tests of the MeasurementType class and constant instances.
    """

    def setUp(self):
        self.types = [m for m in measurement.__dict__.values() if isinstance(m, measurement.MeasurementType)]


    def test_uniqueness(self):
        # Trying to instantiate a duplicate MeasurementType should return the original instance
        FOO = measurement.MeasurementType("Foo", "foo")
        BAR = measurement.MeasurementType("Foo", "foo")
        self.assertEqual(FOO, BAR)
        self.assertIs(FOO, BAR)

        # Again, this time testing against one predefined in the module
        ACCEL = measurement.MeasurementType("Acceleration", "acc")
        self.assertEqual(ACCEL, measurement.ACCELERATION)
        self.assertIs(ACCEL,  measurement.ACCELERATION)


    def test_comp(self):
        # Test that all predefined MeasurementTypes are equal to their string equivalents
        for mt in self.types:
            self.assertEqual(mt, str(mt), "{} not equal to string '{}'".format(repr(mt), mt))


    def test_strings(self):
        # Adding MeasurementTypes concatenates their strings
        self.assertEqual(measurement.ACCELERATION + measurement.PRESSURE,
                         "{} {}".format(measurement.ACCELERATION, measurement.PRESSURE))

        # Negating creates a string prefixed by ``"-"``
        self.assertEqual(-measurement.ACCELERATION,
                         "-{}".format(measurement.ACCELERATION))

        self.assertEqual(measurement.ACCELERATION - measurement.PRESSURE,
                         "{} -{}".format(measurement.ACCELERATION, measurement.PRESSURE))



class GetByTypeTests(unittest.TestCase):
    """ Test the functions that retrieve `Channel` and/or `SubChannel` objects by
        measurement type from an IDE file (`idelib.dataset.Dataset`).
    """

    def setUp(self):
        self.dataset = importFile(os.path.join(os.path.dirname(__file__), "test.ide"))


    def test_get_measurement_type(self):
        self.assertEqual(measurement.get_measurement_type(self.dataset.channels[32][0]),
                         measurement.ACCELERATION)
        self.assertListEqual(measurement.get_measurement_type(self.dataset.channels[80]),
                            [measurement.ACCELERATION]*3)


    def test_split_types(self):
        # XXX: Implement test_split_types
        warnings.warn("measurement.split_types() test not implemented")


    def test_filter_channels(self):
        # XXX: Implement test_filter_channels
        warnings.warn("measurement.filter_channels() test not implemented")


    def test_get_channels(self):
        everything = measurement.get_channels(self.dataset)

        accels = measurement.get_channels(self.dataset, measurement.ACCELERATION)
        self.assertEqual(len(accels), 6)

        noaccel = measurement.get_channels(self.dataset, -measurement.ACCELERATION)
        self.assertEqual(len(noaccel), len(everything) - len(accels))


class ChannelTableFormattingTests(unittest.TestCase):
    """ Test the individual column value formatting functions.
    """

    def test_format_channel_id(self):
        # XXX: Implement test_format_channel_id
        warnings.warn("info.format_channel_id() test not implemented")


    def test_format_timedelta(self):
        # XXX: Implement test_format_timedelta
        warnings.warn("info.format_timedelta() test not implemented")


    def test_format_timestamp(self):
        # XXX: Implement test_format_timestamp
        warnings.warn("info.format_timestamp() test not implemented")


class ChannelTableTests(unittest.TestCase):
    """ Test the "channel table" generating functionality
    """

    def test_get_channel_table(self):
        # XXX: Implement test_get_channel_table
        warnings.warn("info.get_channel_table() test not implemented")


if __name__ == '__main__':
    unittest.main()
