from idelib.importer import importFile
import pytest
from endaq.ide.info import get_channels
from endaq.ide.util import get_accelerometer_info
import os

IDE_FILENAME = os.path.join(os.path.dirname(__file__), "test.ide")


@pytest.fixture
def test_IDE():
    with importFile(IDE_FILENAME) as ds:
        yield ds


def test_get_accelerometer_info(test_IDE):
    """
    Tests highlighted by a comment above the assert.
    `get_accelerometer_bounds` is also tested in this method.
    """
    for accel in get_channels(test_IDE, "accel", False):
        
        sensor_info = get_accelerometer_info(accel)
        l = sensor_info['low_cutoff']
        h = sensor_info['high_cutoff']

        #lower cutoff is always lower than higher cutoff
        assert l < h
        #cutoffs are both positive. By transitivity h > 0 if this test passes.
        assert l > 0
        #rating is one of the valid existing types 
        assert sensor_info['rating'] in [8, 16, 25, 100, 200, 500, 2000, 6000]

def test_incorrect_sensor_get_accelerometer_info(test_IDE):
    """
    tests that passing in non accelerometer info results in a fail.
    """
    accels = get_channels(test_IDE, "accel", False)
    for _, channel in test_IDE.channels.items():
        if channel not in accels:
            with pytest.raises(ValueError):
                get_accelerometer_info(channel)