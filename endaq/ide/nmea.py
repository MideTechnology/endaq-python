import pandas as pd
import datetime as dt
# import endaq.ide.files  # used in commented-out testing at bottom of file
import endaq.ide.measurement as measure
from pynmeagps import NMEAReader, NMEAMessage


def get_nmea(buffer, raw=False):
    """ Simple NMEA sentence extractor.

        :param buffer: a bytearray containing bytes from an IDE dataset
        :param raw: if true, return raw NMEA bytes as opposed to parsed objects. Defaults to false
        :returns: a list of extracted sentences and the leftover buffer (which could have partial sentences completed
            in later blocks).
    """
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


def get_nmea_sentence(dataset, raw=False):
    """ Processes an IDE file to return the parsed data.
        Raise error if there is no raw NMEA channel
        TODO - Need metadata from david to add error raise

        :param dataset: a dataset, as from endaq.ide.files.get_doc()
        :param raw: if true, dump raw nmea sentences as opposed to parsed objects. Defaults to false
        :return: NMEA sentence as NMEAMessage objects or bytes

        :seealso: https://www.sparkfun.com/datasheets/GPS/NMEA%20Reference%20Manual-Rev2.1-Dec07.pdf - Ch. 1. Relevant
            codes for this project are GLL and VTG.

    """
    # fill element list with the needed data (given by david)
    nmeaSentences = []
    rawBuffer = bytearray()
    nmea_channel_id = None
    for chid, ch, in dataset.channels.items():
        if "NMEA" in ch.name:
            nmea_channel_id = chid
            break

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
    include, _ = measure.split_types(measure_type)

    # NOTE: DIRECTION IS A "FUTURE" TYPE IN THE measurement.py FILE. MIGHT WANT TO PULL IT OUT
    # the measure types that make sense to pull from GPS Data are
    # # ANY, DIRECTION, LOCATION, SPEED
    # from the potential future types,
    # # GENERIC, ALTITUDE
    # do not use,
    # # orientation - actually for quaternion stuff
    # # time - used for internal non-nmea gps stuff

    fulldates = []  # we always want the timestamps
    block = []      # holds NMEA messages before processing
    rows = []       # holds all rows, used in dataframe construction

    # checks measurement types
    wantDirs = wantLocs = wantSpds = False
    if measure.ANY in include:
        wantDirs = wantLocs = wantSpds = True
        colnames = ["direction", "latitude", "longitude", "speed", "quality"]
    else:
        colnames = []
        if measure.DIRECTION in include:
            wantDirs = True
            colnames.append("direction")
        if measure.LOCATION in include:
            wantLocs = True
            colnames.append("latitude")
            colnames.append("longitude")
        if measure.SPEED in include:
            wantSpds = True
            colnames.append("speed")
        colnames.append("quality")

    # Column indexes for database construction
    numCol = 0
    dirCol = latCol = lonCol = spdCol = -1
    if wantDirs:
        dirCol = numCol  # in degrees, degree sign character is \u00B0
        numCol += 1
    if wantLocs:
        latCol = numCol  # in ddmm.mmmm
        lonCol = latCol + 1
        numCol += 2
    if wantSpds:
        spdCol = numCol  # in km/h
        numCol += 1
    qualCol = numCol
    numCol += 1

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
                    if wantLocs:    # Lat/Lon Collection
                        # latitude
                        if message.lat:
                            row[latCol] = (-float(message.lat)) if message.NS == 'S' else float(message.lat)
                        # longitude (copy of above)
                        if message.lon:
                            row[lonCol] = (-float(message.lon)) if message.EW == 'W' else float(message.lon)
                elif message.msgID == "GSV":
                    if quality == -1:  # protects quality from multiple GSV messages
                        quality = int(message.numSV)
            if quality >= filter_level:
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


# MAIN
# ds = endaq.ide.files.get_doc("C:\\Users\\jpolischuk\\Downloads\\DAQ12497_000032.IDE")
# nmea_data = get_nmea_sentence(ds)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
# velo_dataframe = get_nmea_measurement(nmea_data, "any", 6)
# print(velo_dataframe)
