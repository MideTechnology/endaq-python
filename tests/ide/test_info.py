from datetime import datetime, timedelta
import os.path
import unittest

import pytest
import numpy as np
from idelib.importer import importFile
import pandas as pd

from endaq.ide import files, info


IDE_FILENAME = os.path.join(os.path.dirname(__file__), "test.ide")


@pytest.fixture
def test_IDE():
    with importFile(IDE_FILENAME) as ds:
        yield ds


class TimeParseTest(unittest.TestCase):
    """ Test the time parsing function.
    """
    def test_parse_time(self):
        """ Test basic handling of non-strings. """
        self.assertEqual(info.parse_time(None), None)
        self.assertEqual(info.parse_time(""), None)
        self.assertEqual(info.parse_time(42), 42)
        self.assertRaises(TypeError, lambda *x: info.parse_time([1]),
                          "parse_time() accepted a bad type (list)")


    def test_parse_time_str(self):
        """ Test time string parsing. """
        self.assertEqual(info.parse_time("11"), 11 * 10**6)
        self.assertEqual(info.parse_time("22:11"), 1331000000)
        self.assertEqual(info.parse_time("3:22:11"), 12131000000)
        self.assertEqual(info.parse_time("1d 3:22:11"), 98531000000)
        self.assertRaises(ValueError, lambda *x: info.parse_time("bogus"),
                          "parse_time() accepted a bad string")


    def test_parse_time_datetime(self):
        """ Test `datetime.datetime` and `datetime.timedelta` parsing. """
        t1 = datetime(2021, 7, 8, 11, 28, 40, 800752)
        t2 = datetime(2021, 7, 7, 0, 0)
        td = timedelta(days=1, seconds=2345, microseconds=6789)

        self.assertEqual(info.parse_time(td), td.total_seconds() * 10**6)
        self.assertEqual(info.parse_time(t1 - t2), 127720.800752 * 10**6)
        self.assertEqual(info.parse_time(t1, t2), 127720.800752 * 10**6)


class ChannelTableFormattingTests(unittest.TestCase):
    """ Test the individual column value formatting functions.
    """

    def test_format_channel_id(self):
        dataset = importFile(IDE_FILENAME)
        self.assertEqual(info.format_channel_id(dataset.channels[59]), '59.*')
        self.assertEqual(info.format_channel_id(dataset.channels[59][0]), '59.0')

        with pytest.warns(UserWarning):
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
        with pytest.warns(UserWarning):
            self.assertEqual(info.format_timedelta(None), "None")


    def test_format_timestamp(self):
        for i in range(0, 10000, 123):
            self.assertTrue(info.format_timestamp(i).startswith(str(i)))
            self.assertTrue(info.format_timestamp(str(i)).startswith(str(i)))

        with pytest.warns(UserWarning):
            self.assertEqual(info.format_timestamp('bogus'), 'bogus')
        with pytest.warns(UserWarning):
            self.assertEqual(info.format_timestamp(None), "None")


class ChannelTableTests(unittest.TestCase):
    """ Test the "channel table" generating functionality
    """
    def setUp(self):
        self.dataset = importFile(IDE_FILENAME)

    def test_get_channel_table(self):
        # TODO: Implement additional get_channel_table() tests?
        ct = info.get_channel_table(self.dataset)

        self.assertEqual(len(ct.data), len(self.dataset.getPlots()),
                         "Length of table's data did not match number of subchannels in IDE")


    def test_channel_table_ranges(self):
        # Exclude channels with extremely low sample rates
        # TODO: change acceptable delta based on sample rate
        MIN_RATE = 10  # Hz
        DELTA = 0.01  # seconds

        # Test with a specified start
        ct1 = info.get_channel_table(self.dataset, start="2s")
        for n, t in enumerate(ct1.data['start']):
            if ct1.data['rate'][n] < MIN_RATE:
                continue
            self.assertAlmostEqual(t / 10**6, 2, delta=DELTA, msg=f"{ct1.data['channel'][n]}")

        # Test with a specified end
        ct2 = info.get_channel_table(self.dataset, end="10s")
        for n, t in enumerate(ct2.data['end']):
            if ct1.data['rate'][n] < MIN_RATE:
                continue
            self.assertAlmostEqual(t / 10**6, 10, delta=DELTA, msg=f"{ct2.data['channel'][n]}")

        # Test with both start and end specified
        ct3 = info.get_channel_table(self.dataset, start="2s", end="10s")
        self.assertListEqual(list(ct3.data['start']), list(ct1.data['start']))
        self.assertListEqual(list(ct3.data['end']), list(ct2.data['end']))


class PrimarySensorTest(unittest.TestCase):
    """
    Tests for the `get_primary_sensor_data()` function.
    """

    def test_get_primary_sensor_data(self):
        doc = files.get_doc(url='https://info.endaq.com/hubfs/data/All-Channels.ide')
        data = info.get_primary_sensor_data(doc=doc)
        self.assertEqual(data.shape, (944990, 4))
        self.assertTrue(isinstance(data.axes[0][0], pd.Timestamp),
                        msg="Incorrect time mode ({!r}, not Timestamp)".format(type(data.axes[0][0])))

        data2 = info.get_primary_sensor_data(doc=doc, measurement_type='accel', time_mode='seconds')
        self.assertTrue(isinstance(data2.axes[0][0], float),
                        msg="Incorrect time mode ({!r}, not float seconds)".format(type(data.axes[0][0])))
        self.assertAlmostEqual(data2.axes[0][0], 0.338775, 5,
                               msg="Incorrect time value")

        data3 = info.get_primary_sensor_data(doc=doc, measurement_type='accel', time_mode='seconds', criteria='duration')
        self.assertAlmostEqual(data3.axes[0][0], 0.341217, 5,
                               msg="Incorrect time value after sorting")

        data4 = info.get_primary_sensor_data(doc=doc, measurement_type='temp')
        self.assertEqual(data4.axes[1][0], 'Control Pad Temperature',
                         msg="Wrong measurement type")

        data5 = info.get_primary_sensor_data(doc=doc, least=True)
        self.assertNotEqual(data.axes[0][0], data5.axes[0][0],
                            msg="Primary sensor data returned as least sensor data")


@pytest.mark.parametrize("time_mode, subchannel", [
    ("seconds", False),
    ("timedelta", False),
    ("datetime", False),
    ("seconds", True),
    ("timedelta", True),
    ("datetime", True),
])
def test_to_pandas(test_IDE, time_mode, subchannel):
    channel = test_IDE.channels[32]
    if subchannel:
        channel = channel.subchannels[0]
    eventarray = channel.getSession()

    result = info.to_pandas(channel, time_mode=time_mode)

    assert len(result) == len(eventarray)
    if subchannel:
        assert result.columns.tolist() == [channel.name]
    else:
        assert result.columns.tolist() == [sch.name for sch in channel.subchannels]
    assert np.issubdtype(
        result.index.values.dtype,
        dict(seconds=np.float64, timedelta=np.timedelta64, datetime=np.datetime64)[time_mode],
    )
    assert np.all(result.to_numpy() == eventarray.arrayValues().T)


def test_to_pandas_tz(test_IDE):
    """ Minimal test of time zones in `to_pandas()`. Mostly a sanity check. """
    channel = test_IDE.channels[32]
    
    result_utc = info.to_pandas(channel, tz="utc")
    result_local = info.to_pandas(channel, tz="local")
    result_device = info.to_pandas(channel, tz="device")

    assert "UTC" in str(result_utc.index.dtype)
    assert "device" in str(result_device.index.dtype)


if __name__ == '__main__':
    unittest.main()
