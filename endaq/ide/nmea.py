import pandas as pd
import datetime as dt
# import endaq.ide.files  # used in commented-out testing at bottom of file
import endaq.ide.measurement as measure
from pynmeagps import NMEAReader, NMEAMessage
from idelib.dataset import Dataset


def get_nmea(buffer: bytearray, raw=False):
    """ Simple NMEA sentence extractor.

        :param buffer: a bytearray containing bytes from an IDE dataset
        :param raw: if true, return raw NMEA bytes as opposed to parsed objects. Defaults to false
        :returns: a list of extracted sentences and the leftover buffer (which could have partial sentences completed
            in later blocks).
    """
    if not isinstance(buffer, bytearray):
        raise ValueError(f"Bad type for buffer: {type(buffer)}")
    sentences = []
    while b'\r\n' in buffer:
        chunk, _delimiter, buffer = buffer.partition(b'\r\n')
        _junk, prefix, chunk = chunk.partition(b'$G')
        if chunk:
            if raw:
                sentences.append(prefix + chunk + _delimiter)
            else:
                sentences.append(NMEAReader.parse(bytes(prefix + chunk + _delimiter)))
    return sentences, buffer


def get_nmea_sentence(dataset: [Dataset], channel = None, raw=False):
    """ Processes an IDE file to return the parsed data.
        Raise error if there is no raw NMEA channel

        :param dataset: a dataset, as from endaq.ide.files.get_doc()
        :param channel: allows user to specify dataset channel
        :param raw: if true, dump raw nmea sentences as opposed to parsed objects. Defaults to false
        :return: NMEA sentence as NMEAMessage objects or bytes

        :seealso: https://www.sparkfun.com/datasheets/GPS/NMEA%20Reference%20Manual-Rev2.1-Dec07.pdf - Ch. 1. Relevant
            codes for this project are GLL and VTG.

    """
    if not isinstance(dataset, Dataset):
        raise ValueError("Requires Dataset object as input")
    # fill element list with the needed data (given by david)
    nmeaSentences = []
    rawBuffer = bytearray()
    nmea_channel_id = None

    if channel is None:  # find it
        for chid, ch, in dataset.channels.items():
            if "NMEA" in ch.name:
                nmea_channel_id = chid
                break
    else:  # given it
        nmea_channel_id = channel
    if nmea_channel_id is None:  # check there's something
        raise ValueError("No raw NMEA channel specified or found")

    for el in dataset.ebmldoc:
        if el.name == "ChannelDataBlock":
            if el[0].value == nmea_channel_id:
                rawBuffer.extend(el.dump()["ChannelDataPayload"])
                extractData, rawBuffer = get_nmea(rawBuffer, raw)
                nmeaSentences.extend(extractData)

    return nmeaSentences


def get_nmea_measurement(data: [NMEAMessage], measure_type, filter_level: int = 0, timestamp="GPS") -> pd.DataFrame:
    """ Returned dataframe format should appear like dataframes for other channels run through the
        endaq.ide.to_pandas function.

        :param data: A dataset or list of NMEAMessage Objects. The data to pull measurements from
        :param measure_type: Determines the kind of measurements that will be returned in the dataframe. Can accept
            ANY (returns all following measurements), DIRECTION (degrees, True North), LOCATION (lat/long),
            and SPEED (km/h).
        :param filter_level: reliability filter level - Will only process data when connected to this number of
        satellites. Defaults to 0, maximum of 12.
        :param timestamp - Default is to provide timestamp from the GPS message. Selecting between GPS time and device
            time will be the last piece of the job.
    """
    if not isinstance(data, list) or not isinstance(data[0], NMEAMessage):
        raise ValueError("Data expected as NMEAMessage objects. Try .get_nmea_sentence()")
    include, _ = measure.split_types(measure_type)
    fulldates = []  # we always want the timestamps
    block = []      # holds NMEA messages before processing
    rows = []       # holds all rows, used in dataframe construction

    if measure.ANY in include:
        # indexes for data columns
        numCol = 5
        # checks measurement types
        wantDirs = wantLocs = wantSpds = True
        dirCol, latCol, lonCol, spdCol, qualCol = range(numCol)
        colnames = ["direction", "latitude", "longitude", "speed", "quality"]
    else:
        numCol = 0
        wantDirs = wantLocs = wantSpds = False
        dirCol = latCol = lonCol = spdCol = -1
        colnames = []
        if measure.DIRECTION in include:
            wantDirs = True
            dirCol = numCol  # in degrees, degree sign character is \u00B0
            numCol += 1
            colnames.append("direction")
        if measure.LOCATION in include:
            wantLocs = True
            latCol = numCol  # in ddmm.mmmm
            lonCol = latCol + 1
            numCol += 2
            colnames.append("latitude")
            colnames.append("longitude")
        if measure.SPEED in include:
            wantSpds = True
            spdCol = numCol  # in km/h
            numCol += 1
            colnames.append("speed")
        qualCol = numCol     # number of satellites in use
        numCol += 1
        colnames.append("quality")

    # NOTE: messages containing more than timestamps start appearing at time 17:17:44 in test data file
    collecting = False  # signifies that we're building the block
    processing = False  # signifies that we're processing the block
    for sentence in data:
        if collecting:
            if sentence.msgID == "GLL":  # Final message in block
                collecting = False  # switch off for final conditional
                processing = True  # begin processing previous block
            block.append(sentence)

        if processing:
            quality = -1
            row = [None] * numCol   # holds one row, used to store data before filter_level validation
            timestamp = None
            for message in block:   # any updates please follow below format
                if message.msgID == "RMC":
                    timestamp = dt.datetime.combine(message.date, message.time)
                elif message.msgID == "VTG":
                    if wantDirs:    # Direction Collection
                        if message.cogt:  # this is false if cogt = ""
                            row[dirCol] = float(message.cogt)
                    if wantSpds:    # Speed Collection
                        if message.sogk:
                            row[spdCol] = float(message.sogk)
                elif message.msgID == "GGA":
                    quality = int(message.numSV)
                    if wantLocs:    # Lat/Lon Collection
                        # latitude
                        if message.lat:
                            row[latCol] = float(message.lat)  # NMEAMessage automatically adjusts sign
                        # longitude (copy of above)
                        if message.lon:
                            row[lonCol] = float(message.lon)

            if quality >= filter_level and any(c is not None for c in row):  # quality row and not all none
                fulldates.append(timestamp)
                row[qualCol] = quality
                rows.append(row)
            processing = False  # processing complete, switch off for final conditional
            block.clear()
        # this if avoids any data with no timestamp information
        if sentence.msgID == "RMC" and collecting is not True and processing is not True:
            if sentence.time == "" or sentence.date == "":
                continue
            else:
                collecting = True
                block.append(sentence)

    return pd.DataFrame(data=rows, index=fulldates, columns=colnames)


# TESTING
# ds = endaq.ide.files.get_doc("C:\\Users\\jpolischuk\\Downloads\\DAQ12497_000032.IDE")
# nmea_data = get_nmea_sentence(ds)
# for i in range(len(nmea_data)):
#     # if i in range(944, 954):
#     print(str(i) + ": " + repr(nmea_data[i]))
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
# velo_dataframe = get_nmea_measurement(nmea_data, "any", 8)
# print(velo_dataframe)
