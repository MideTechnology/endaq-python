from __future__ import annotations

from dataclasses import dataclass
import typing
from typing import List, Tuple, Optional
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


MPS2_TO_G = 1 / 9.80665
MPS_TO_MMPS = 1000
MPS_TO_UMPS = 10 ** 6
MPS_TO_KMPH = 3600 / 1e3
M_TO_MM = 1000


@dataclass
class CalcParams:
    """
    The parameters for configuring the calculation routines in
    `CalcCache`.

    Each of these parameters is *intentionally* left w/o a default value.
    Instead, defaults are provided at the function signatures for the
    `endaq.batch.core` functions. This ensures that data is passed correctly
    to them from the `CalcParam` object.
    """

    accel_highpass_cutoff: Optional[float]
    accel_integral_tukey_percent: float
    accel_integral_zero: typing.Literal["start", "mean", "median"]
    accel_start_time: Optional[np.timedelta64]
    accel_end_time: Optional[np.timedelta64]
    accel_start_margin: Optional[np.timedelta64]
    accel_end_margin: Optional[np.timedelta64]
    psd_freq_bin_width: float
    psd_window: str
    pvss_init_freq: float
    pvss_bins_per_octave: float
    vc_init_freq: float
    vc_bins_per_octave: float


class CalcCache:
    """
    A wrapper for `idelib.dataset.Dataset` that caches channel data streams as
    `pandas.Dataset`s.
    """

    PV_NATURAL_FREQS = np.logspace(0, 12, base=2, num=12 * 12 + 1, endpoint=True)

    def __init__(self, data, params: CalcParams):
        """
        Copies out the numpy arrays for the highest priority channel for each
        sensor type, and any relevant metadata.  Cuts them into chunks.
        """
        if (
            params.accel_start_time is not None
            and params.accel_start_margin is not None
        ):
            raise ValueError(
                "only one of `accel_start_time` and `accel_start_margin` may be set at once"
            )
        if params.accel_end_time is not None and params.accel_end_margin is not None:
            raise ValueError(
                "only one of `accel_end_time` and `accel_end_margin` may be set at once"
            )

        self._channels = data
        self._params = params

    @classmethod
    def from_ide(cls, dataset, params: CalcParams, preferred_chs: List[int] = []):
        """
        Instantiate a new `CalcCache` object from an IDE file.
        """
        data = ide_utils.dict_chs_best(
            (
                (utype, ch_struct)
                for (utype, ch_struct) in ide_utils.chs_by_utype(dataset)
                if len(ch_struct.eventarray) > 0
            ),
            max_key=lambda x: (x.channel.id in preferred_chs, len(x.eventarray)),
        )

        return cls(data, params=params)

    @dataclass
    class InputDataWrapper:
        data: pd.DataFrame
        units: Tuple[str, str]

        def to_pandas(self, time_mode="datetime"):
            expected_index_types = dict(
                timedelta=(pd.TimedeltaIndex, "TimedeltaIndex"),
                datetime=(pd.DatetimeIndex, "DatetimeIndex"),
                seconds=(
                    (pd.Float64Index, pd.Int64Index, pd.UInt64Index, pd.RangeIndex),
                    "{Float64/Int64/UInt64/Range}Index",
                ),
            )
            if not isinstance(self.data.index, expected_index_types[time_mode][0]):
                raise ValueError(
                    f"expected '{time_mode}' data index to be of type"
                    f"`{expected_index_types[time_mode][1]}`, "
                    f"instead found {type(self.data.index)}"
                )

            self.data.index.name = "timestamp"
            self.data.columns.name = "axis"

            return self.data

    @classmethod
    def from_raw_data(
        cls, data: List[Tuple[pd.DataFrame, Tuple[str, str]]], params: CalcParams
    ):
        """
        Instantiate a new `CalcCache` object from raw DataFrame / metadata pairs.
        """
        data = {
            ide_utils.UTYPE_GROUPS[units[0]]: cls.InputDataWrapper(data, units)
            for (data, units) in data
        }
        return cls(data, params=params)

    # ==========================================================================
    # Data Processing, just to make init cleaner
    # ==========================================================================

    @cached_property
    def _accelerationData(self):
        """Populate the _acceleration* fields, including splitting and extending data."""
        ch_struct = self._channels.get("acc", None)
        if ch_struct is None:
            warnings.warn(f"no acceleration channel in data")
            return pd.DataFrame(
                np.empty((0, 3), dtype=float),
                index=pd.Series([], dtype="timedelta64[ns]", name="time"),
                columns=pd.Series(["X", "Y", "Z"], name="axis"),
            )

        aUnits = ch_struct.units[1]
        try:
            conversionFactor = {  # core units = m/s^2
                "g": 1 / MPS2_TO_G,
                "m/s\u00b2": 1,
            }[aUnits.lower()]
        except KeyError:
            raise ValueError(f'unknown acceleration channel units "{aUnits}"')

        aData = conversionFactor * ch_struct.to_pandas(time_mode="timedelta")
        dt = endaq.calc.sample_spacing(aData, convert=None)

        if self._params.accel_start_margin is not None:
            margin = int(np.ceil(self._params.accel_start_margin / dt))
            aData = aData.iloc[margin:]
        elif self._params.accel_start_time is not None:
            aData = aData.loc[self._params.accel_start_time :]
        if self._params.accel_end_margin is not None:
            margin = int(np.ceil(self._params.accel_end_margin / dt))
            aData = aData.iloc[: (-margin or None)]
        elif self._params.accel_end_time is not None:
            aData = aData.loc[: self._params.accel_end_time]

        aData = filters.butterworth(
            aData, low_cutoff=self._params.accel_highpass_cutoff
        )

        assert isinstance(aData, pd.DataFrame)
        return aData

    @cached_property
    def _accelerationResultantData(self):
        return pd.Series(
            stats.L2_norm(self._accelerationData.to_numpy(), axis=1),
            index=self._accelerationData.index,
        ).to_frame()

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

        return ch_struct.to_pandas(time_mode="timedelta")

    @cached_property
    def _velocityData(self):
        aData = self._accelerationData
        if aData.size == 0:
            return aData

        if not self._params.accel_highpass_cutoff:
            warnings.warn(
                "no highpass filter used before integration; "
                "velocity calculation may be unstable"
            )

        return integrate._integrate(
            filters.butterworth(
                aData,
                low_cutoff=self._params.accel_highpass_cutoff,
                tukey_percent=self._params.accel_integral_tukey_percent,
            ),
            zero=self._params.accel_integral_zero,
        )

    @cached_property
    def _velocityResultantData(self):
        return pd.Series(
            stats.L2_norm(self._velocityData.to_numpy(), axis=1),
            index=self._velocityData.index,
        ).to_frame()

    @cached_property
    def _displacementData(self):
        vData = self._velocityData
        if vData.size == 0:
            return vData

        if not self._params.accel_highpass_cutoff:
            warnings.warn(
                "no highpass filter used before integration; "
                "displacement calculation may be unstable"
            )

        return integrate._integrate(
            filters.butterworth(
                vData,
                low_cutoff=self._params.accel_highpass_cutoff,
                tukey_percent=self._params.accel_integral_tukey_percent,
            ),
            zero=self._params.accel_integral_zero,
        )

    @cached_property
    def _displacementResultantData(self):
        return pd.Series(
            stats.L2_norm(self._displacementData.to_numpy(), axis=1),
            index=self._displacementData.index,
        ).to_frame()

    @cached_property
    def _PVSSData(self):
        aData = self._accelerationData
        if aData.size == 0:
            pvss = aData.copy()
            pvss.index.name = "frequency (Hz)"
            return pvss

        freqs = endaq.calc.logfreqs(
            aData, self._params.pvss_init_freq, self._params.pvss_bins_per_octave
        )
        freqs = freqs[
            (freqs >= 1 / (endaq.calc.sample_spacing(aData) * aData.shape[0]))
        ]
        pv = shock.shock_spectrum(
            aData,
            freqs,
            damp=0.05,
            mode="pvss",
        )

        return pv

    @cached_property
    def _PVSSResultantData(self):
        aData = self._accelerationData
        if aData.size == 0:
            pvss = self._accelerationResultantData.copy()
            pvss.index.name = "frequency (Hz)"
            return pvss

        freqs = endaq.calc.logfreqs(
            aData, self._params.pvss_init_freq, self._params.pvss_bins_per_octave
        )
        freqs = freqs[
            (freqs >= 1 / (endaq.calc.sample_spacing(aData) * aData.shape[0]))
        ]
        pv = shock.shock_spectrum(
            aData,
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
            psdData = aData.copy()
            psdData.index.name = "frequency (Hz)"
            return psdData

        return endaq.calc.psd.welch(
            aData,
            bin_width=self._params.psd_freq_bin_width,
            window=self._params.psd_window,
            average="median",
        )

    @cached_property
    def _PSDResultantData(self):
        return self._PSDData.sum(axis="columns").to_frame()

    @cached_property
    def _VCCurveData(self):
        """Calculate Vibration Criteria (VC) Curves for the accelerometer."""
        psdData = self._PSDData
        if psdData.size == 0:
            vcData = psdData.copy()
            vcData.index.name = "frequency (Hz)"
            return vcData

        return psd.vc_curves(
            self._PSDData,
            fstart=self._params.vc_init_freq,
            octave_bins=self._params.vc_bins_per_octave,
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

        return conversionFactor * ch_struct.to_pandas(time_mode="timedelta")

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

        return conversionOffset + conversionFactor * ch_struct.to_pandas(
            time_mode="timedelta"
        )

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

        data = ch_struct.to_pandas(time_mode="timedelta")
        dt = endaq.calc.sample_spacing(data, convert="to_seconds")
        units = ch_struct.units[1]
        if units.lower() == "q":
            quat_raw = data[["W", "X", "Y", "Z"]].to_numpy()

            data = pd.DataFrame(
                (180 / np.pi) * quat.quat_to_angvel(quat_raw, dt, qaxis=1),
                index=data.index,
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
                data, prefix_len=max(4, int(np.ceil(0.25 / dt)))
            )
        elif units.lower() not in ("dps", "deg/s"):
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

        # resampling destroys last values -> no resampling
        return ch_struct.to_pandas(time_mode="timedelta")

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

        return MPS_TO_KMPH * ch_struct.to_pandas(time_mode="timedelta")

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
        return MPS2_TO_G * rms

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
        return MPS_TO_MMPS * rms

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
        return M_TO_MM * rms

    @cached_property
    def accPeakFull(self):
        """Peak instantaneous tri-axial acceleration"""
        max_abs = self._accelerationData.abs().max(axis="rows")
        max_abs_res = self._accelerationResultantData.max(axis="rows")
        max_abs["Resultant"] = max_abs_res.iloc[0]
        max_abs.name = "Peak Absolute Acceleration"
        return MPS2_TO_G * max_abs

    @cached_property
    def pseudoVelPeakFull(self):
        """Peak Pseudo Velocity"""
        max_pv = self._PVSSData.max(axis="rows")
        max_pv_res = self._PVSSResultantData.max(axis="rows")
        max_pv["Resultant"] = max_pv_res.iloc[0]
        max_pv.name = "Peak Pseudo Velocity Shock Spectrum"
        return MPS2_TO_G * max_pv

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
