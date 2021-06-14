from datetime import timedelta
import os.path
import unittest
import warnings

from idelib.importer import importFile
from endaq.ide import info, measurement


IDE_FILENAME = os.path.join(os.path.dirname(__file__), "test.ide")

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
        self.dataset = importFile(IDE_FILENAME)


    def test_get_measurement_type(self):
        self.assertEqual(measurement.get_measurement_type(self.dataset.channels[32][0]),
                         measurement.ACCELERATION)
        self.assertListEqual(measurement.get_measurement_type(self.dataset.channels[80]),
                            [measurement.ACCELERATION]*3)


    def test_split_types(self):
        # XXX: Implement test_split_types
        inc, exc = measurement.split_types("*")
        warnings.warn("measurement.split_types() test not implemented")


    def test_filter_channels(self):
        """ Test test_filter_channels() filtering of Channels (filter applies
            if any SubChannel matches).
        """
        channels = self.dataset.channels
        everything = measurement.filter_channels(channels)
        self.assertListEqual(everything, measurement.filter_channels(list(channels.values())),
                             "filter_channels(list) did not match filter_channels(dict)")

        accels = measurement.filter_channels(channels, measurement.ACCELERATION)
        self.assertEqual(len(accels), 2)

        noaccel = measurement.filter_channels(channels, -measurement.ACCELERATION)
        self.assertEqual(len(noaccel), len(everything) - len(accels))


    def test_filter_channels_subchannels(self):
        """ Test test_filter_channels() filtering of SubChannels.
        """
        subchannels = self.dataset.getPlots()
        everything = measurement.filter_channels(subchannels)

        accels = measurement.filter_channels(subchannels, measurement.ACCELERATION)
        self.assertEqual(len(accels), 6)

        noaccel = measurement.filter_channels(subchannels, -measurement.ACCELERATION)
        self.assertEqual(len(noaccel), len(everything) - len(accels))


    def test_get_channels(self):
        everything = measurement.get_channels(self.dataset)

        accels = measurement.get_channels(self.dataset, measurement.ACCELERATION)
        self.assertEqual(len(accels), 6)

        noaccel = measurement.get_channels(self.dataset, -measurement.ACCELERATION)
        self.assertEqual(len(noaccel), len(everything) - len(accels))

        everything = measurement.get_channels(self.dataset, subchannels=False)

        accels = measurement.get_channels(self.dataset, measurement.ACCELERATION, subchannels=False)
        self.assertEqual(len(accels), 2)

        noaccel = measurement.get_channels(self.dataset, -measurement.ACCELERATION, subchannels=False)
        self.assertEqual(len(noaccel), len(everything) - len(accels))


class ChannelTableFormattingTests(unittest.TestCase):
    """ Test the individual column value formatting functions.
    """

    def test_format_channel_id(self):
        dataset = importFile(IDE_FILENAME)
        self.assertEqual(info.format_channel_id(dataset.channels[59]), '59.*')
        self.assertEqual(info.format_channel_id(dataset.channels[59][0]), '59.0')

        self.assertEqual(info.format_channel_id(None), "None")


    def test_format_timedelta(self):
        # Note: only the start of strings is checked in order to avoid
        # differences in selected number of significant digits
        td = timedelta(seconds=0)
        self.assertTrue(info.format_timedelta(td).startswith('00:00.'))

        td = timedelta(seconds=1623430749.8969631)
        self.assertTrue(info.format_timedelta(td).startswith('18789d 16:59:09.'))

        # Number instead of timedelta. Unlikely but not not impossible.
        self.assertTrue(info.format_timedelta(100000000).startswith('01:40.'))
        self.assertEqual(info.format_timedelta(None), "None")


    def test_format_timestamp(self):
        for i in range(0, 10000, 123):
            self.assertTrue(info.format_timestamp(i).startswith(str(i)))
            self.assertTrue(info.format_timestamp(str(i)).startswith(str(i)))

        self.assertEqual(info.format_timestamp('bogus'), 'bogus')
        self.assertEqual(info.format_timestamp(None), "None")


class ChannelTableTests(unittest.TestCase):
    """ Test the "channel table" generating functionality
    """
    def setUp(self):
        self.dataset = importFile(IDE_FILENAME)

    def test_get_channel_table(self):
        # XXX: Implement additional get_channel_table() tests
        ct = info.get_channel_table(self.dataset)

        self.assertEqual(len(ct.data), len(self.dataset.getPlots()),
                         "Length of table's data did not match number of subchannels in IDE")


if __name__ == '__main__':
    unittest.main()
