from collections import defaultdict, namedtuple
import warnings

from endaq.ide import to_pandas


class NoChannelException(Exception):
    pass


class ChannelStruct(namedtuple("ChannelStruct", "channel, sch_ids")):
    @property
    def eventarray(self):
        return self.channel.getSession()

    @property
    def fs(self):
        return self.eventarray.getSampleRate() if len(self.eventarray) > 0 else None

    @property
    def units(self):
        return self.channel[self.sch_ids[0]].units

    @property
    def axis_names(self):
        return [self.channel.subchannels[i].name for i in self.sch_ids]

    def to_pandas(self, *args, **kwargs):
        df = to_pandas(self.channel, *args, **kwargs).iloc[:, self.sch_ids]
        df.columns.name = "axis"
        return df


def chs_by_utype(dataset):
    """Group subchannels together by channel & utype."""
    for ch_id, channel in dataset.channels.items():
        # Group subchannel id's by unit type
        sch_ids_by_utype = defaultdict(list)
        for sch_id, subchannel in enumerate(channel.subchannels):
            sch_ids_by_utype[subchannel.units[0]].append(sch_id)

        # Separate each channel into subchannels of like-unit-type
        for utype, sch_ids in sch_ids_by_utype.items():
            yield utype, ChannelStruct(channel, sch_ids)


def map_utypes(iterable, utypes_map):
    """Re-label utype-schs pairs with new utypes."""
    return ((utypes_map.get(utype, utype), ch_struct) for utype, ch_struct in iterable)


def dict_groups(iterable):
    """Group an iterable of key-value pairs into a dict of lists of items."""
    result = defaultdict(list)
    for k, v in iterable:
        result[k].append(v)
    return result


def dict_chs_best(iterable, max_key=lambda x: len(x.eventarray)):
    """
    Group an iterable of utype-ch_struct pairs into a dict, keeping only those
    channels with the highest sample length for their utype.
    """
    result = {}
    for utype, ch_struct in iterable:
        if not utype in result:
            result[utype] = ch_struct
            continue

        result[utype] = max(result[utype], ch_struct, key=max_key)

    return result


def get_ch_type_best(dataset, utype, max_key=lambda x: len(x.eventarray)):
    """Get the highest sample-length acceleration channel from a recording."""
    chs = chs_by_utype(dataset)
    utype_chs = (ch_struct for (ut, ch_struct) in chs if ut == utype)

    try:
        return max(utype_chs, key=max_key)
    except ValueError:
        raise NoChannelException(f'no channels of type "{utype}" in recording')
