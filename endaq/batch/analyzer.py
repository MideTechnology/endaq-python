import logging
import sys
import warnings

if sys.version_info[:2] >= (3, 8):
    from functools import cached_property
else:
    from backports.cached_property import cached_property


import numpy as np
import pandas as pd

import endaq.calc
from endaq.calc import psd, stats, shock, integrate, filters

from endaq.batch import quat
from endaq.batch.utils import ide_utils


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
            return pd.DataFrame(
                np.empty((0, 3), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["X", "Y", "Z"], name="axis"),
            )

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

        aData = conversionFactor * ch_struct.to_pandas(time_mode="timedelta")

        if self._accel_start_margin is not None:
            margin = int(
                np.ceil(
                    ch_struct.fs * self._accel_start_margin / np.timedelta64(1, "s")
                )
            )
            aData = aData.iloc[margin:]
        elif self._accel_start_time is not None:
            aData = aData.loc[self._accel_start_time :]
        if self._accel_end_margin is not None:
            margin = int(
                np.ceil(ch_struct.fs * self._accel_end_margin / np.timedelta64(1, "s"))
            )
            aData = aData.iloc[: (-margin or None)]
        elif self._accel_end_time is not None:
            aData = aData.loc[: self._accel_end_time]

        aData = filters.butterworth(aData, low_cutoff=self._accel_highpass_cutoff)

        assert isinstance(aData, pd.DataFrame)
        return aData

    @cached_property
    def _accelerationResultantData(self):
        return self._accelerationData.apply(stats.L2_norm, axis="columns").to_frame()

    @cached_property
    def _microphoneData(self):
        """Populate the _microphone* fields, including splitting and extending data."""
        ch_struct = self._channels.get("mic", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 1), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["mic"], name="axis"),
            )

        units = ch_struct.units[1]
        if units.lower() != "a":
            raise ValueError(f'unknown microphone channel units "{units}"')

        self._micName = ch_struct.channel.name
        self._micFs = ch_struct.fs
        data = ch_struct.to_pandas(time_mode="timedelta")

        return data

    @cached_property
    def _velocityData(self):
        aData = self._accelerationData
        if aData.size == 0:
            return aData

        if not self._accel_highpass_cutoff:
            logging.warning(
                "no highpass filter used before integration; "
                "velocity calculation may be unstable"
            )

        return integrate._integrate(aData)

    @cached_property
    def _displacementData(self):
        vData = self._velocityData
        if vData.size == 0:
            return vData

        if not self._accel_highpass_cutoff:
            logging.warning(
                "no highpass filter used before integration; "
                "displacement calculation may be unstable"
            )

        return integrate._integrate(vData)

    @cached_property
    def _PVSSData(self):
        aData = self._accelerationData
        if aData.size == 0:
            pvss = aData[:]
            pvss.index.name = "frequency (Hz)"
            return pvss

        freqs = endaq.calc.logfreqs(
            aData, self._pvss_init_freq, self._pvss_bins_per_octave
        )
        freqs = freqs[
            (freqs >= self._accelerationFs / self._accelerationData.shape[0])
        ]
        pv = shock.shock_spectrum(
            self._accelerationData,
            freqs,
            damp=0.05,
            mode="pvss",
        )

        return pv

    @cached_property
    def _PVSSResultantData(self):
        aData = self._accelerationData
        if aData.size == 0:
            pvss = self._accelerationResultantData[:]
            pvss.index.name = "frequency (Hz)"
            return pvss

        freqs = endaq.calc.logfreqs(
            aData, self._pvss_init_freq, self._pvss_bins_per_octave
        )
        freqs = freqs[
            (freqs >= self._accelerationFs / self._accelerationData.shape[0])
        ]
        pv = shock.shock_spectrum(
            self._accelerationData,
            freqs,
            damp=0.05,
            mode="pvss",
            aggregate_axes=True,
        )

        return pv

    @cached_property
    def _PSDData(self):
        aData = self._accelerationData
        if aData.size == 0:
            psdData = aData[:]
            psdData.index.name = "frequency (Hz)"
            return psdData

        return endaq.calc.psd.welch(
            aData,
            bin_width=self._psd_freq_bin_width,
            window=self._psd_window,
            average="median",
        )

    @cached_property
    def _VCCurveData(self):
        """Calculate Vibration Criteria (VC) Curves for the accelerometer."""
        psdData = self._PSDData
        if psdData.size == 0:
            vcData = psdData[:]
            vcData.index.name = "frequency (Hz)"
            return vcData

        return psd.vc_curves(
            self._PSDData,
            fstart=self._vc_init_freq,
            octave_bins=self._vc_bins_per_octave,
        )

    @cached_property
    def _pressureData(self):
        """Populate the _pressure* fields, including splitting and extending data."""
        ch_struct = self._channels.get("pre", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 1), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["Pressure"], name="axis"),
            )

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
        data = conversionFactor * ch_struct.to_pandas(time_mode="timedelta")

        return data

    @cached_property
    def _temperatureData(self):
        """Populate the _temperature* fields, including splitting and extending data."""
        ch_struct = self._channels.get("tmp", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 1), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["Temperature"], name="axis"),
            )

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
        data = conversionOffset + conversionFactor * ch_struct.to_pandas(
            time_mode="timedelta"
        )

        return data

    @cached_property
    def _gyroscopeData(self):
        """Populate the _gyro* fields, including splitting and extending data."""
        ch_struct = self._channels.get("gyr", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 3), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["X", "Y", "Z"], name="axis"),
            )

        self._gyroName = ch_struct.channel.name
        self._gyroFs = ch_struct.fs

        units = ch_struct.units[1]
        if units.lower() == "q":
            quat_df = ch_struct.to_pandas(time_mode="timedelta")
            quat_raw = quat_df[["W", "X", "Y", "Z"]].to_numpy()

            data = pd.DataFrame(
                (180 / np.pi)
                * quat.quat_to_angvel(quat_raw, 1 / self._gyroFs, qaxis=1),
                index=quat_df.index,
                columns=pd.Series(["X", "Y", "Z"], name="axis"),
            )

            def strip_invalid_prefix(data, prefix_len):
                """Search prefix for invalid data and remove it (if any)."""
                data_array = data.to_numpy()
                data_mag = stats.L2_norm(data_array[: 4 * prefix_len], axis=-1)
                # the derivative method for `quat_to_angvel` uses the *average*
                # of adjacent differences
                # -> any rotation spikes will result in two adjacent,
                # nearly-equal peaks
                data_agg = 0.5 * (data_mag[:-1] + data_mag[1:])
                argmax_prefix = np.argmax(data_agg[:prefix_len])

                # Prefix data is considered "anomalous" if it is much larger
                # than any surrounding data
                if data_agg[argmax_prefix] > 2 * data_mag[prefix_len:].max():
                    data = data.iloc[argmax_prefix + 2 :]

                return data

            data = strip_invalid_prefix(
                data, prefix_len=max(4, int(np.ceil(0.25 * self._gyroFs)))
            )
        elif units.lower() in ("dps", "deg/s"):
            data = ch_struct.to_pandas(time_mode="timedelta")
        else:
            raise ValueError(f'unknown gyroscope channel units "{units}"')

        return data

    @cached_property
    def _gpsPositionData(self):
        ch_struct = self._channels.get("gps", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 2), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["Latitude", "Longitude"], name="axis"),
            )

        units = ch_struct.units[1]
        if units.lower() != "degrees":
            raise ValueError(f'unknown GPS position channel units "{units}"')

        data = ch_struct.to_pandas(time_mode="timedelta")

        self._gpsName = ch_struct.channel.name
        self._gpsFs = ch_struct.fs
        # resampling destroys last values -> no resampling

        return data

    @cached_property
    def _gpsSpeedData(self):
        ch_struct = self._channels.get("spd", None)
        if ch_struct is None:
            return pd.DataFrame(
                np.empty((0, 1), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["GPS Speed"], name="axis"),
            )

        units = ch_struct.units[1]
        if units != "m/s":
            raise ValueError(f'unknown GPS ground speed channel units "{units}"')

        data = self.MPS_TO_KMPH * ch_struct.to_pandas(time_mode="timedelta")

        self._gpsSpeedName = ch_struct.channel.name
        self._gpsSpeedFs = ch_struct.fs

        return data

    # ==========================================================================
    # Analyses
    # ==========================================================================

    @cached_property
    def accRMSFull(self):
        """Accelerometer Tri-axial RMS."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = self._accelerationData.apply(
                stats.rms, axis="rows", raw=True
            )  # RuntimeWarning: Mean of empty slice.

        rms["Resultant"] = stats.L2_norm(rms.to_numpy())
        rms.name = "RMS Acceleration"
        return self.MPS2_TO_G * rms

    @cached_property
    def velRMSFull(self):
        """Velocity Tri-axial RMS, after applying a 0.1Hz highpass filter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = self._velocityData.apply(
                stats.rms, axis="rows", raw=True
            )  # RuntimeWarning: Mean of empty slice.

        rms["Resultant"] = stats.L2_norm(rms.to_numpy())
        rms.name = "RMS Velocity"
        return self.MPS_TO_MMPS * rms

    @cached_property
    def disRMSFull(self):
        """Displacement Tri-axial RMS, after applying a 0.1Hz highpass filter."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = self._displacementData.apply(
                stats.rms, axis="rows", raw=True
            )  # RuntimeWarning: Mean of empty slice.

        rms["Resultant"] = stats.L2_norm(rms.to_numpy())
        rms.name = "RMS Displacement"
        return self.M_TO_MM * rms

    @cached_property
    def accPeakFull(self):
        """Peak instantaneous tri-axial acceleration"""
        max_abs = self._accelerationData.abs().max(axis="rows")
        max_abs_res = self._accelerationResultantData.max(axis="rows")
        max_abs["Resultant"] = max_abs_res.iloc[0]
        max_abs.name = "Peak Absolute Acceleration"
        return self.MPS2_TO_G * max_abs

    @cached_property
    def pseudoVelPeakFull(self):
        """Peak Pseudo Velocity"""
        max_pv = self._PVSSData.max(axis="rows")
        max_pv_res = self._PVSSResultantData.max(axis="rows")
        max_pv["Resultant"] = max_pv_res.iloc[0]
        max_pv.name = "Peak Pseudo Velocity Shock Spectrum"
        return self.MPS2_TO_G * max_pv

    @cached_property
    def gpsLocFull(self):
        """Average GPS location"""
        data = self._gpsPositionData
        # 0's occur when gps doesn't have a "lock" -> remove them
        data = data.iloc[np.all(data.to_numpy() != 0, axis=1)]
        if data.size == 0:
            gpsPos = pd.Series(
                [np.nan, np.nan],
                index=pd.Series(["Latitude", "Longitude"], name="axis"),
            )
        else:
            gpsPos = data.iloc[-1]

        gpsPos.name = "GPS Position"
        return gpsPos

    @cached_property
    def gpsSpeedFull(self):
        """Average GPS speed"""
        data = self._gpsSpeedData
        # 0's occur when gps doesn't have a "lock" -> remove them
        data = data.iloc[data.to_numpy() != 0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gpsSpeed = data.mean()  # RuntimeWarning: Mean of empty slice.

        gpsSpeed.name = "GPS Speed"
        return gpsSpeed

    @cached_property
    def gyroRMSFull(self):
        """Gyroscope RMS"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rms = self._gyroscopeData.apply(
                stats.rms, axis="rows", raw=True
            )  # RuntimeWarning: Mean of empty slice.
        rms["Resultant"] = stats.L2_norm(rms.to_numpy())
        rms.name = "RMS Angular Velocity"
        return rms

    @cached_property
    def micRMSFull(self):
        """Microphone RMS"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mic = self._microphoneData.apply(
                stats.rms, axis="rows", raw=True
            )  # RuntimeWarning: Mean of empty slice.

        mic.name = "RMS Microphone"
        return mic

    @cached_property
    def tempFull(self):
        """Average Temperature"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp = self._temperatureData.mean()  # RuntimeWarning: Mean of empty slice.

        temp.name = "Average Temperature"
        return temp

    @cached_property
    def pressFull(self):
        """Average Pressure"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            press = self._pressureData.mean()  # RuntimeWarning: Mean of empty slice.

        press.name = "Average Temperature"
        return press
