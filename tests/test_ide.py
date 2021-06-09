import unittest

from endaq.ide import info, measurement


class MeasurementTypeTests(unittest.TestCase):

    def setUp(self):
        self.types = [m for m in measurement.__dict__.values() if isinstance(m, measurement.MeasurementType)]


    def test_uniqueness(self):
        # Trying to instantiate a duplicate MeasurementType should return the original instance
        FOO = measurement.MeasurementType("Foo", "foo", labels=("foo",))
        BAR = measurement.MeasurementType("Foo", "foo", labels=("foo",))
        self.assertEqual(FOO, BAR)
        self.assertIs(FOO, BAR)

        # Again, this time testing against one predefined in the module
        ACCEL = measurement.MeasurementType("Acceleration", "acc", labels=())
        self.assertEqual(ACCEL, measurement.ACCELERATION)
        self.assertIs(ACCEL,  measurement.ACCELERATION)


    def test_comp(self):
        # Test that all predefined MeasurementTypes are equal to their string equivalents
        for mt in self.types:
            self.assertEqual(mt, str(mt))


    def test_query(self):
        # Adding MeasurementTypes concatenates their strings
        self.assertEqual(measurement.ACCELERATION + measurement.PRESSURE,
                         "{} {}".format(measurement.ACCELERATION, measurement.PRESSURE))
        # Negating creates a string prefixed by ``"-"``
        self.assertEqual(-measurement.ACCELERATION,
                         "-{}".format(measurement.ACCELERATION))



if __name__ == '__main__':
    unittest.main()
