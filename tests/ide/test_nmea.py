from endaq.ide.files import get_doc
from pynmeagps import NMEAMessage
import pandas
import datetime
import pytest
import endaq.ide.nmea as nmea
import endaq.ide.measurement as measure
import os.path


@pytest.fixture()
def junk():
    """ A short string of garbage bytes to use for testing unclean data"""
    return bytearray(b'hsdfufuf')


@pytest.fixture()
def ide_doc():
    """The IDE document to use for testing the parser"""
    return get_doc(os.path.join(os.path.dirname(__file__), "nmea.IDE"))


@pytest.fixture()
def single_msg():
    """A singluar NMEA message, in bytes"""
    return bytearray(b'$GNRMC,171800.00,A,4229.72940,N,07108.37734,W,0.305,,100622,,,A*7D\r\n')


@pytest.fixture
def multi_msg():
    """A 'block' of NMEA messages, taken from the nmea.IDE doc"""
    return (bytearray(b'$GNRMC,171800.00,A,4229.72940,N,07108.37734,W,0.305,,100622,,,A*7D\r\n')
            + bytearray(b'$GNVTG,,T,,M,0.305,N,0.565,K,A*3D\r\n')
            + bytearray(b'$GNGGA,171800.00,4229.72940,N,07108.37734,W,1,05,2.88,209.6,M,-33.1,M,,*76\r\n')
            + bytearray(b'$GNGSA,A,3,17,24,14,,,,,,,,,,5.23,2.88,4.36*1E\r\n')
            + bytearray(b'$GNGSA,A,3,69,75,,,,,,,,,,,5.23,2.88,4.36*16\r\n')
            + bytearray(b'$GPGSV,2,1,07,02,,,28,11,,,25,12,,,24,14,21,166,35*46\r\n')
            + bytearray(b'$GPGSV,2,2,07,17,60,087,36,24,07,276,36,28,,,25*7E\r\n')
            + bytearray(b'$GLGSV,2,1,05,67,07,018,17,68,33,062,23,69,29,123,35,75,33,311,20*60\r\n')
            + bytearray(b'$GLGSV,2,2,05,85,21,281,22*55\r\n')
            + bytearray(b'$GNGLL,4229.72940,N,07108.37734,W,171800.00,A,A*65\r\n')
            )


def test_get_nmea(junk, single_msg, multi_msg):
    # nothing and junk
    assert(nmea.get_nmea(bytearray(), raw=True)) == ([], bytearray())
    assert(nmea.get_nmea(junk, raw=True)) == ([], junk)
    assert(nmea.get_nmea(bytearray(), raw=False)) == ([], bytearray())
    assert(nmea.get_nmea(junk, raw=False)) == ([], junk)
    # bad data
    with pytest.raises(ValueError) as excInfo1:
        nmea.get_nmea(None)
    with pytest.raises(ValueError) as excInfo2:
        nmea.get_nmea("hello")
    assert(excInfo1.type is ValueError)
    assert(excInfo2.type is ValueError)
    # single message
    assert(nmea.get_nmea(single_msg, raw=True) == ([bytearray(b'$GNRMC,171800.00,A,4229.72940,N,07108.37734,W,0.305,,100622,,,A*7D\r\n')], bytearray()))
    sentences, buffer = nmea.get_nmea(single_msg, raw=False)
    assert(str(sentences) == str([NMEAMessage('GN','RMC', 0, payload=['171800.00', 'A', '4229.72940', 'N', '07108.37734', 'W', '0.305', '', '100622', '', '', 'A'])]))
    # multi message
    sentences, buffer = nmea.get_nmea(multi_msg, raw=False)
    rsentences, rbuffer = nmea.get_nmea(multi_msg, raw=True)
    assert(len(sentences) == 10)
    assert(len(rsentences) == 10)
    assert(str(rsentences[0]) == str(single_msg))
    #junk'd message
    _junkMM = junk + multi_msg + junk
    jsentences, jbuffer = nmea.get_nmea(_junkMM, raw=False)
    jrsentences, jrbuffer = nmea.get_nmea(_junkMM, raw=True)
    assert(str(jbuffer) == str(junk))
    assert(str(jrbuffer) == str(junk))


def test_get_nmea_sentence(junk, ide_doc):
    # bad data
    with pytest.raises(ValueError) as excInfo1:
        nmea.get_nmea_sentences(None)
    with pytest.raises(ValueError) as excInfo2:
        nmea.get_nmea_sentences("hello")
    assert(excInfo1.type is ValueError)
    assert(excInfo2.type is ValueError)
    # single line
    dataset = nmea.get_nmea_sentences(ide_doc)
    rdataset = nmea.get_nmea_sentences(ide_doc, raw=True)
    cdataset = nmea.get_nmea_sentences(ide_doc, channel=100)
    rcdataset = nmea.get_nmea_sentences(ide_doc, channel=100, raw=True)
    assert(str(dataset[944]) == str(NMEAMessage('GN','RMC', 0,
                                    payload=['171800.00', 'A', '4229.72940', 'N', '07108.37734', 'W', '0.305', '',
                                             '100622', '', '', 'A'])))
    # multi line
    assert(len(dataset) == 1384)
    assert(len(rdataset) == 1384)
    assert(len(cdataset) == 1384)
    assert(len(rcdataset) == 1384)


def test_get_nmea_measurement(junk, ide_doc):
    # bad data
    with pytest.raises(ValueError) as excInfo1:
        nmea.get_nmea_measurement(None, measure.ANY)
    with pytest.raises(ValueError) as excInfo2:
        nmea.get_nmea_measurement("hello", measure.ANY)
    with pytest.raises(ValueError) as excInfo3:
        nmea.get_nmea_measurement([], measure.ANY)
    # passing raw data
    itsBloodyRaw = nmea.get_nmea_sentences(ide_doc, raw=True)
    with pytest.raises(ValueError) as excInfo4:
        nmea.get_nmea_measurement(itsBloodyRaw, measure.ANY)
    assert(excInfo1.type is ValueError)
    assert(excInfo2.type is ValueError)
    assert(excInfo3.type is ValueError)
    assert(excInfo4.type is ValueError)
    #passing dataset and sentences
    timestamps = [datetime.datetime(2022,6,10,17,18,s) for s in range(39, 44)]
    singledf = pandas.DataFrame({
        "speed": [0.228, 0.136, 0.146, 0.186, 0.072],
        "quality": [8] * 5
    }, index=timestamps)
    fulldf = pandas.DataFrame({
        "direction": [None]*5,
        "latitude": [42.495848,42.495865,42.495874,42.495881,42.495888],
        "longitude": [-71.138997,-71.138970,-71.138958,-71.138947,-71.138935],
        "speed": [0.228,0.136,0.146,0.186,0.072],
        "quality": [8]*5
    }, index=timestamps)
    testsingledoc = nmea.get_nmea_measurement(ide_doc, measure.SPEED, 8)
    testfulldoc = nmea.get_nmea_measurement(ide_doc, measure.ANY, 8)
    sentences = nmea.get_nmea_sentences(ide_doc)
    testsingledf = nmea.get_nmea_measurement(sentences, measure.SPEED, 8)
    testfulldf = nmea.get_nmea_measurement(sentences, measure.ANY, 8)
    # the below *ARE* tests
    # need approx because float stuff, need dict conversion because pandas isn't supported
    _ = testsingledoc.to_dict("list") == pytest.approx(singledf.to_dict("list"))
    _ = testfulldoc.to_dict("list") == pytest.approx(fulldf.to_dict("list"))
    _ = testsingledf.to_dict("list") == pytest.approx(singledf.to_dict("list"))
    _ = testfulldf.to_dict("list") == pytest.approx(fulldf.to_dict("list"))




