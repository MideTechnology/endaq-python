import os.path
import unittest

from idelib.importer import importFile
from endaq.ide import measurement


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
            self.assertEqual(mt, str(mt), f"{mt!r} not equal to string '{mt}'")


    def test_strings(self):
        # Adding MeasurementTypes concatenates their strings
        self.assertEqual(measurement.ACCELERATION + measurement.PRESSURE,
                         f"{measurement.ACCELERATION} {measurement.PRESSURE}")

        # Negating creates a string prefixed by ``"-"``
        self.assertEqual(-measurement.ACCELERATION,
                         f"-{measurement.ACCELERATION}")

        self.assertEqual(measurement.ACCELERATION - measurement.PRESSURE,
                         f"{measurement.ACCELERATION} -{measurement.PRESSURE}")



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
        inc, exc = measurement.split_types("*")
        self.assertEqual(len(measurement.MeasurementType.names), len(inc))
        self.assertEqual(len(exc), 0)

        inc, exc = measurement.split_types(measurement.ACCELERATION + measurement.PRESSURE - measurement.LIGHT)
        self.assertIn(measurement.ACCELERATION, inc)
        self.assertIn(measurement.PRESSURE, inc)
        self.assertIn(measurement.LIGHT, exc)
        self.assertTrue(measurement.LIGHT not in inc)
        self.assertTrue(measurement.PRESSURE not in exc)


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


if __name__ == '__main__':
    unittest.main()
