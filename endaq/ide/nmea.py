import datetime

import pandas as pd
import datetime as dt

import endaq.ide.files
import endaq.ide.measurement as measure
from ebmlite import loadSchema
from pynmeagps import NMEAReader, NMEAMessage, NMEAParseError


def get_nmea(buffer, raw=False):
    """ Simple NMEA sentence extractor. Returns a list of extracted
        sentences and the leftover buffer (which could have partial
        sentences completed in later blocks).
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

        ::seealso:
            https://www.sparkfun.com/datasheets/GPS/NMEA%20Reference%20Manual-Rev2.1-Dec07.pdf - Ch. 1. Relevant
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
        :param measure_type: To be decided by David
        :param filter_level: reliability filter level - Will only process data when connected to this number of
        satelites. Defaults to 0, maximum of 12.
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

    block = []      # holds NMEA messages before processing
    fulldates = []  # we always want the timestamps
    rows = []       # holds all rows, used in dataframe construction

    # checks measurement types. future use to help set column indexes?
    wantDirs = measure.DIRECTION in include or measure.ANY
    wantLocs = measure.LOCATION in include or measure.ANY
    wantSpds = measure.SPEED in include or measure.ANY

    # NOTES: messages containing more than timestamps start appearing at time 17:17:44
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
            row = []  # holds one row, used to store data before filter_level validation
            timestamp = None
            for message in block:
                if message.msgID == "RMC":
                    timestamp = dt.datetime.combine(message.date, message.time)
                if message.msgID == "GSV":
                    if quality == -1:  # protects quality from multiple GSV messages
                        quality = int(message.numSV)
                if wantDirs and message.msgID == "VTG":  # Direction Collection
                    if message.cogt == "":
                        row.append(float(0.0))
                    else:
                        row.append(float(message.cog))
                if wantLocs and message.msgID == "GGA":  # Location Collection
                    # latitude
                    if message.lat == "":
                        row.append("0000.0000")
                    else:
                        if message.NS == "S":
                            row.append(-float(message.lat))
                        else:
                            row.append(float(message.lat))
                    # longitude (copy of above)
                    if message.lon == "":
                        row.append("0000.0000")
                    else:
                        if message.EW == "W":
                            row.append(-float(message.lon))
                        else:
                            row.append(float(message.lon))
                if wantSpds and message.msgID == "VTG":  # Speed Collection
                    if message.sogk == "":
                        row.append(float(0.0))
                    else:
                        row.append(float(message.sogk))
            if quality >= filter_level:
                fulldates.append(timestamp)
                row.append(quality)
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

    colnames = ["direction", "speed", "latitude", "longitude", "quality"]
    # direction: degrees (unicode char \u00B0)
    return pd.DataFrame(data=rows, index=fulldates, columns=colnames)


# MAIN
ds = endaq.ide.files.get_doc("C:\\Users\\jpolischuk\\Downloads\\DAQ12497_000032.IDE")
nmea_data = get_nmea_sentence(ds)
# for i in range(len(nmea_data)):
#     print(str(i) + ": " + str(nmea_data[i]))
# for i in range(3):
#     print("\n\tSPACER\n")
# for i in range(len(nmea_data)):
#     if nmea_data[i].msgID in ["RMC", "GSV"]:
#         print(str(i) + ": " + nmea_data[i].msgID + " found: " + repr(nmea_data[i]))
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
velo_dataframe = get_nmea_measurement(nmea_data, measure.ANY, 6)
print(velo_dataframe)
print(velo_dataframe.shape)

