from functools import wraps
import logging
import sys
import warnings

if sys.version_info[:2] >= (3, 8):
    from functools import cached_property
else:
    from backports.cached_property import cached_property


import numpy as np
import scipy.signal
import pandas as pd

from endaq.batch import quat
from endaq.batch.utils import ide_utils
from endaq.batch.utils.calc import psd, stats, shock, integrate, filters


def as_series(
    unit_type,
    data_name,
    *,
    edit_axis_names=lambda axis_names: axis_names,
    default_axis_names,
):
    """Format method output as a pandas Series."""

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            data = method(self, *args, **kwargs)

            try:
                ch_struct = self._channels[unit_type]
            except KeyError:
                axis_names = default_axis_names
            else:
                axis_names = edit_axis_names(ch_struct.axis_names)

            return pd.Series(
                data,
                index=pd.Index(axis_names, name="axis"),
                name=data_name,
            )

        return wrapper

    return decorator


class Analyzer:
    """
    A class which will run the analyses for the endaq cloud work.  Take the
    important info from the IDE document so that it can be released from memory.
    """

    # TODO: This needs to be reworked, it's currently not very efficient.
    #       -
    #       The current strategy is to:
    #         1. Create a list of time indices for each segment, where the
    #            beginning of the first segment is the earliest data present
    #            among all channels (not al channels start and end at the same time)
    #         2. Create a numpy array from each channel, and grab metadata such
    #            as sampling frequency.
    #         3. For channels with a low sampling frequency, resample them to
    #            five times the segment frequency.
    #         4. Pad each channel with nans so that data exists for each channel
    #            in each time segment.
    #         5. Split each of these by the times created in step 1
    #         6. Lazily run analyses as they're called and cache results.
    #       -
    #       The issue is that this is somewhat memory intensive and just doesn't
    #       need to be. I think we can more efficiently do piecewise evaluation
    #       and re-use intermediate calculations without saving all of the file
    #       as an array.
    #       _
    #       Proposed strategy:
    #         1. Same as above, create a list of time indices for each segment,
    #            where the beginning of the first segment is the earliest data
    #            present among all channels (not al channels start and end at
    #            the same time)
    #         2. For each sensor type:
    #              a. If the sensor is not present, create a list of NaNs in the
    #                 correct shape.
    #              b. For sensors with low sampling rates, use some sort of
    #                 wrapper or generator that will still lazily load data from
    #                 the file but will return interpolated data as if it had a
    #                 higher sample rate (five times the segment frequency)
    #              c. Iterate through the segments.  If the segment is out of
    #                 range, skip it and place Nans in each analysis for that
    #                 channel type.
    #              d. Do the analyses for that segment.  This will allow us to
    #                 reuse intermediate calculations, such as the FFT of the
    #                 acceleration.  This is used to calculate the RMS for the
    #                 acceleration and its integrals.
    #

    MPS2_TO_G = 1 / 9.80665
    MPS_TO_MMPS = 1000
    MPS_TO_UMPS = 10 ** 6
    MPS_TO_KMPH = 3600 / 1e3
    M_TO_MM = 1000

    PV_NATURAL_FREQS = np.logspace(0, 12, base=2, num=12 * 12 + 1, endpoint=True)

    def __init__(
        self,
        doc,
        *,
        preferred_chs=[],
        accel_highpass_cutoff,
        accel_start_time,
        accel_end_time,
        accel_start_margin,
        accel_end_margin,
        psd_freq_bin_width,
        psd_window="hanning",
        pvss_init_freq,
        pvss_bins_per_octave,
        vc_init_freq,
        vc_bins_per_octave,
    ):
        """
        Copies out the numpy arrays for the highest priority channel for each
        sensor type, and any relevant metadata.  Cuts them into chunks.
        """
        if accel_start_time is not None and accel_start_margin is not None:
            raise ValueError(
                "only one of `accel_start_time` and `accel_start_margin` may be set at once"
            )
        if accel_end_time is not None and accel_end_margin is not None:
            raise ValueError(
                "only one of `accel_end_time` and `accel_end_margin` may be set at once"
            )

        self._channels = ide_utils.dict_chs_best(
            (
                (utype, ch_struct)
                for (utype, ch_struct) in ide_utils.chs_by_utype(doc)
                if len(ch_struct.eventarray) > 0
            ),
            max_key=lambda x: (x.channel.id in preferred_chs, len(x.eventarray)),
        )

        self._filename = doc.filename
        self._accelerationFs = None  # gets set in `_accelerationData`
        self._accel_highpass_cutoff = accel_highpass_cutoff
        self._accel_start_time = accel_start_time
        self._accel_end_time = accel_end_time
        self._accel_start_margin = accel_start_margin
        self._accel_end_margin = accel_end_margin
        self._psd_window = psd_window
        self._psd_freq_bin_width = psd_freq_bin_width
        self._pvss_init_freq = pvss_init_freq
        self._pvss_bins_per_octave = pvss_bins_per_octave
        self._vc_init_freq = vc_init_freq
        self._vc_bins_per_octave = vc_bins_per_octave

    # ==========================================================================
    # Data Processing, just to make init cleaner
    # ==========================================================================

    @cached_property
    def _accelerationData(self):
        """Populate the _acceleration* fields, including splitting and extending data."""
        ch_struct = self._channels.get("acc", None)
        if ch_struct is None:
            logging.warning(f"no acceleration channel in {self._filename}")
            return np.empty((3, 0), dtype=np.float)

        aUnits = ch_struct.units[1]
        try:
            conversionFactor = {  # core units = m/s^2
                "g": 1 / self.MPS2_TO_G,
                "m/s\u00b2": 1,
            }[aUnits.lower()]
        except KeyError:
            raise ValueError(f'unknown acceleration channel units "{aUnits}"')

        self._accelerationName = ch_struct.channel.name
        self._accelerationFs = ch_struct.fs

        times = ch_struct.eventarray.arraySlice()[0] * 1e-6  # us -> s
        aData = conversionFactor * ch_struct.eventarray.arrayValues(
            subchannels=ch_struct.sch_ids,
        )

        if self._accel_start_margin is not None:
            margin = int(np.ceil(ch_struct.fs * self._accel_start_margin))
            aData = aData[:, margin:]
        elif self._accel_start_time is not None:
            i = np.searchsorted(times, self._accel_start_time)
            aData = aData[:, i:]
        if self._accel_end_margin is not None:
            margin = int(np.ceil(ch_struct.fs * self._accel_end_margin))
            aData = aData[:, : (-margin or None)]
        elif self._accel_end_time is not None:
            i = np.searchsorted(times, self._accel_end_time)
            aData = aData[:, :i]

        if self._accel_highpass_cutoff:
            aData = filters.highpass(
                aData, fs=ch_struct.fs, cutoff=self._accel_highpass_cutoff, axis=-1
            )

        return aData

    @cached_property
    def _accelerationResultant(self):
        return stats.L2_norm(self._accelerationData, axis=0)

    @cached_property
    def _microphoneData(self):
        """Populate the _microphone* fields, including splitting and extending data."""
        ch_struct = self._channels.get("mic", None)
        if ch_struct is None:
            return np.empty(0, dtype=np.float)

        units = ch_struct.units[1]
        if units.lower() != "a":
            raise ValueError(f'unknown microphone channel units "{units}"')

        self._micName = ch_struct.channel.name
        self._micFs = ch_struct.fs
        data = ch_struct.eventarray.arrayValues(subchannels=ch_struct.sch_ids)

        return data

    @cached_property
    def _velocityData(self):
        aData = self._accelerationData
        if aData.size == 0:
            return np.empty((3, 0), dtype=np.float)

        if not self._accel_highpass_cutoff:
            logging.warning(
                "no highpass filter used before integration; "
                "velocity calculation may be unstable"
            )

        vData = integrate._integrate(aData, dt=1 / self._accelerationFs, axis=1)

        return vData

    @cached_property
    def _displacementData(self):
        vData = self._velocityData
        if vData.size == 0:
            return np.empty((3, 0), dtype=np.float)

        if not self._accel_highpass_cutoff:
            logging.warning(
                "no highpass filter used before integration; "
                "displacement calculation may be unstable"
            )

        dData = integrate._integrate(vData, dt=1 / self._accelerationFs, axis=1)

        return dData

    @cached_property
    def _PVSSData(self):
        aData = self._accelerationData
        if aData.size == 0:
            return np.empty(0, dtype=np.float), self._accelerationData

        log2_f0 = np.log2(self._pvss_init_freq)
        log2_f1 = np.log2(self._accelerationFs)
        num_bins = np.floor(
            self._pvss_bins_per_octave * (log2_f1 - 1 - log2_f0)
        ).astype(int)

        freqs = np.logspace(
            start=log2_f0,
            stop=log2_f0 + num_bins / self._pvss_bins_per_octave,
            num=num_bins + 1,
            base=2,
            endpoint=True,
        )
        freqs = freqs[
            (freqs >= self._accelerationFs / self._accelerationData.shape[-1])
        ]
        pv = shock.pseudo_velocity(
            self._accelerationData,
            freqs,
            dt=1 / self._accelerationFs,
            damp=0.05,
            two_sided=False,
            axis=-1,
        )
        assert pv.ndim == 2
        assert 1 <= pv.shape[0] <= 3
        assert pv.shape[-1] == len(freqs)

        return freqs, pv

    @cached_property
    def _PSDData(self):
        aData = self._accelerationData
        if aData.size == 0:
            return np.empty(0, dtype=np.float), self._accelerationData

        return scipy.signal.welch(
            aData,
            fs=self._accelerationFs,
            nperseg=int(np.ceil(self._accelerationFs / self._psd_freq_bin_width)),
            window=self._psd_window,
            average="median",
            axis=1,
        )

    @cached_property
    def _VCCurveData(self):
        """Calculate Vibration Criteria (VC) Curves for the accelerometer."""
        aData = self._accelerationData
        if aData.size == 0:
            return np.empty(0, dtype=np.float), self._accelerationData

        """
        Theory behind the calculation:
        
        Let x(t) be a real-valued time-domain signal, and X(2πf) = F{x(t)}(2πf)
        be the Fourier Transform of that signal. By Parseval's Theorem,

            ∫x(t)^2 dt = ∫|X(2πf)|^2 df

        (see https://en.wikipedia.org/wiki/Parseval%27s_theorem#Notation_used_in_physics)

        Rewriting the right side of that equation in the discrete form becomes

            ∫x(t)^2 dt ≈ ∑ |X[k]|^2 • ∆f
        
        where ∆f = fs/N = (1/∆t) / N = 1/T.
        Limiting the right side to a range of discrete frequencies (k_0, k_1):

            ∫x(t)^2 dt ≈ [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f

        The VC curve calculation is the RMS over the time-domain. If T is the
        duration of the time-domain signal, then:

            √((1/T) ∫x(t)^2 dt)
                ≈ √((1/T) [∑; k=k_0 -> k≤k_1] |X[k]|^2 • ∆f)
                = ∆f • √([∑; k=k_0 -> k≤k_1] |X[k]|^2)

        If the time-series data is acceleration, then the signal needs to first
        be integrated into velocity. This can be done in the frequency domain
        by replacing |X(2πf)|^2 with (1/2πf)^2 |X(2πf)|^2.
        """
        f, a_psd = self._PSDData
        f, v_psd = psd.differentiate(f, a_psd, n=-1)
        f_oct, v_psd_oct = psd.to_octave(
            f,
            v_psd,
            fstart=self._vc_init_freq,
            octave_bins=self._vc_bins_per_octave,
            mode="sum",
        )
        v_vc = np.sqrt(f[1] * v_psd_oct)  # the PSD must already scale by ∆f?

        return f_oct, v_vc

    @cached_property
    def _pressureData(self):
        """Populate the _pressure* fields, including splitting and extending data."""
        ch_struct = self._channels.get("pre", None)
        if ch_struct is None:
            return np.empty(0, dtype=np.float)

        units = ch_struct.units[1]
        try:
            conversionFactor = {  # core units = kPa
                "pa": 1e-3,
                "psi": 6.89476,
                "atm": 101.325,
            }[units.lower()]
        except KeyError:
            raise ValueError(f'unknown pressure channel units "{units}"')

        self._preName = ch_struct.channel.name
        self._preFs = ch_struct.fs
        data = conversionFactor * ch_struct.eventarray.arrayValues(
            subchannels=ch_struct.sch_ids,
        )

        return data

    @cached_property
    def _temperatureData(self):
        """Populate the _temperature* fields, including splitting and extending data."""
        ch_struct = self._channels.get("tmp", None)
        if ch_struct is None:
            return np.empty(0, dtype=np.float)

        units = ch_struct.units[1]
        try:
            conversionFactor, conversionOffset = {  # core units = degrees C
                "\xb0c": (1, 0),
                "\xb0k": (1, 273.15),
                "\xb0f": (5 / 9, -32 * (5 / 9)),
            }[units.lower()]
        except KeyError:
            raise ValueError(f'unknown temperature channel units "{units}"')

        self._tmpName = ch_struct.channel.name
        self._tmpFs = ch_struct.fs
        data = conversionOffset + conversionFactor * ch_struct.eventarray.arrayValues(
            subchannels=ch_struct.sch_ids,
        )

        return data

    @cached_property
    def _gyroscopeData(self):
        """Populate the _gyro* fields, including splitting and extending data."""
        ch_struct = self._channels.get("gyr", None)
        if ch_struct is None:
            return np.empty((3, 0), dtype=np.float)

        self._gyroName = ch_struct.channel.name
        self._gyroFs = ch_struct.fs

        units = ch_struct.units[1]
        if units.lower() == "q":
            quat_array = ch_struct.eventarray.arrayValues(subchannels=ch_struct.sch_ids)
            quat_raw = quat_array[
                [3, 0, 1, 2]
            ]  # reorders to <W, X, Y, Z> & strips out the "Acc" channel

            data = (180 / np.pi) * quat.quat_to_angvel(quat_raw.T, 1 / self._gyroFs).T

            def strip_invalid_prefix(data, prefix_len):
                """Search prefix for invalid data and remove it (if any)."""
                data_mag = stats.L2_norm(data[: 4 * prefix_len], axis=0)
                # the derivative method for `quat_to_angvel` uses the *average*
                # of adjacent differences
                # -> any rotation spikes will result in two adjacent,
                # nearly-equal peaks
                data_agg = 0.5 * (data_mag[:-1] + data_mag[1:])
                argmax_prefix = np.argmax(data_agg[:prefix_len])

                # Prefix data is considered "anomalous" if it is much larger
                # than any surrounding data
                if data_agg[argmax_prefix] > 2 * data_mag[prefix_len:].max():
                    data = data[..., argmax_prefix + 2 :]

                return data

            data = strip_invalid_prefix(
                data, prefix_len=max(4, int(np.ceil(0.25 * self._gyroFs)))
            )
        elif units.lower() in ("dps", "deg/s"):
            data = ch_struct.eventarray.arrayValues(subchannels=ch_struct.sch_ids)
        else:
            raise ValueError(f'unknown gyroscope channel units "{units}"')

        return data

    def _processHumidity(self, channel):
        """Populate the _humidity* fields, including splitting and extending data."""
        pass

    @cached_property
    def _gpsPositionData(self):
        ch_struct = self._channels.get("gps", None)
        if ch_struct is None:
            return np.empty((2, 0), dtype=np.float)

        units = ch_struct.units[1]
        if units.lower() != "degrees":
            raise ValueError(f'unknown GPS position channel units "{units}"')

        data = ch_struct.eventarray.arrayValues(subchannels=ch_struct.sch_ids)

        self._gpsName = ch_struct.channel.name
        self._gpsFs = ch_struct.fs
        # resampling destroys last values -> no resampling

        return data

    @cached_property
    def _gpsSpeedData(self):
        ch_struct = self._channels.get("spd", None)
        if ch_struct is None:
            return np.empty(0)

        units = ch_struct.units[1]
        if units != "m/s":
            raise ValueError(f'unknown GPS ground speed channel units "{units}"')

        data = self.MPS_TO_KMPH * ch_struct.eventarray.arrayValues(
            subchannels=ch_struct.sch_ids
        )

        self._gpsSpeedName = ch_struct.channel.name
        self._gpsSpeedFs = ch_struct.fs

        return data

    # ==========================================================================
    # Analyses
    # ==========================================================================

    @cached_property
    @as_series(
        "acc",
        "RMS Acceleration",
        edit_axis_names=lambda axis_names: axis_names + ["Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def accRMSFull(self):
        """Accelerometer Tri-axial RMS."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = stats.rms(
                self._accelerationData, axis=1
            )  # RuntimeWarning: Mean of empty slice.
        return self.MPS2_TO_G * np.append(rms, stats.L2_norm(rms))

    @cached_property
    @as_series(
        "acc",
        "RMS Velocity",
        edit_axis_names=lambda axis_names: axis_names + ["Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def velRMSFull(self):
        """Velocity Tri-axial RMS, after applying a 0.1Hz highpass filter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = stats.rms(
                self._velocityData, axis=1
            )  # RuntimeWarning: Mean of empty slice.
        return self.MPS_TO_MMPS * np.append(rms, stats.L2_norm(rms))

    @cached_property
    @as_series(
        "acc",
        "RMS Displacement",
        edit_axis_names=lambda axis_names: axis_names + ["Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def disRMSFull(self):
        """Displacement Tri-axial RMS, after applying a 0.1Hz highpass filter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = stats.rms(
                self._displacementData, axis=1
            )  # RuntimeWarning: Mean of empty slice.
        return self.M_TO_MM * np.append(rms, stats.L2_norm(rms))

    @cached_property
    @as_series(
        "acc",
        "Peak Absolute Acceleration",
        edit_axis_names=lambda axis_names: axis_names + ["Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def accPeakFull(self):
        """Peak instantaneous tri-axial acceleration"""
        max_abs = stats.max_abs(self._accelerationData, axis=1)
        max_abs_res = np.amax(
            stats.L2_norm(self._accelerationData, axis=0),
            initial=-np.inf,
            axis=-1,
        )
        return self.MPS2_TO_G * np.nan_to_num(
            np.append(max_abs, max_abs_res), nan=np.nan, posinf=np.inf, neginf=np.nan
        )

    @cached_property
    @as_series(
        "acc",
        "Peak Pseudo Velocity Shock Spectrum",
        edit_axis_names=lambda axis_names: axis_names + ["Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def pseudoVelPeakFull(self):
        """Peak Pseudo Velocity"""
        if self._PVSSData[1].size == 0:
            return np.full(self._PVSSData[1].shape[0] + 1, np.nan)

        pv = self._PVSSData[1]
        max_pv = np.amax(pv, initial=-np.inf, axis=1)
        max_pv_res = np.amax(stats.L2_norm(pv, axis=0), initial=-np.inf, axis=-1)
        return self.MPS_TO_MMPS * np.nan_to_num(
            np.append(max_pv, max_pv_res), nan=np.nan, posinf=np.inf, neginf=np.nan
        )

    @cached_property
    @as_series("gps", "GPS Position", default_axis_names=["Latitude", "Longitude"])
    def gpsLocFull(self):
        """Average GPS location"""
        data = self._gpsPositionData
        # 0's occur when gps doesn't have a "lock" -> remove them
        data = data[:, np.all(data != 0, axis=0)]
        if data.size == 0:
            return [np.nan, np.nan]

        return data[..., -1]

    @cached_property
    @as_series("spd", "GPS Speed", default_axis_names=["Ground"])
    def gpsSpeedFull(self):
        """Average GPS speed"""
        data = self._gpsSpeedData
        # 0's occur when gps doesn't have a "lock" -> remove them
        data = data[data != 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean(data)  # RuntimeWarning: Mean of empty slice.

    @cached_property
    @as_series(
        "gyr",
        "RMS Angular Velocity",
        edit_axis_names=lambda axis_names: ["X", "Y", "Z", "Resultant"],
        default_axis_names=["X", "Y", "Z", "Resultant"],
    )
    def gyroRMSFull(self):
        """Gyroscope RMS"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = stats.rms(
                self._gyroscopeData, axis=1
            )  # RuntimeWarning: Mean of empty slice.
        return np.append(rms, stats.L2_norm(rms))

    @cached_property
    @as_series("mic", "RMS Microphone", default_axis_names=[""])
    def micRMSFull(self):
        """Microphone RMS"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats.rms(
                self._microphoneData
            )  # RuntimeWarning: Mean of empty slice.

    @cached_property
    @as_series("tmp", "Average Temperature", default_axis_names=[""])
    def tempFull(self):
        """Average Temperature"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._temperatureData.mean()  # RuntimeWarning: Mean of empty slice.

    @cached_property
    @as_series("pre", "Average Pressure", default_axis_names=[""])
    def pressFull(self):
        """Average Pressure"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._pressureData.mean()  # RuntimeWarning: Mean of empty slice.
