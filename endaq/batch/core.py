from __future__ import annotations

from typing import List

from functools import partial
import warnings
import os

import numpy as np
import pandas as pd

import endaq.ide
from endaq.calc import stats as calc_stats
from endaq.calc import psd as calc_psd

import endaq.batch.analyzer


def _make_meta(dataset):
    """Generate a pandas object containing metadata for the given recording."""
    serial_no = dataset.recorderInfo["RecorderSerial"]
    start_time = np.datetime64(dataset.sessions[0].utcStartTime, "s") + np.timedelta64(
        dataset.sessions[0].firstTime, "us"
    )

    return pd.Series(
        [serial_no, start_time],
        index=["serial number", "start time"],
        name=dataset.filename,
    )


def _make_psd(ch_data_cache, fstart=None, bins_per_octave=None):
    """
    Format the PSD of the main accelerometer channel into a pandas object.

    The PSD is scaled to units of g^2/Hz (g := gravity = 9.80665 meters per
    square second).
    """
    df_psd = ch_data_cache._PSDData
    if df_psd.size == 0:
        return None

    if bins_per_octave is not None:
        df_psd = calc_psd.to_octave(
            df_psd,
            fstart=(fstart or 1),
            octave_bins=bins_per_octave,
            agg=np.mean,
        )

    df_psd["Resultant"] = np.sum(df_psd.to_numpy(), axis=1)
    df_psd = df_psd * endaq.batch.analyzer.MPS2_TO_G ** 2  # (m/s^2)^2/Hz -> g^2/Hz

    return df_psd.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


def _make_pvss(ch_data_cache):
    """
    Format the PVSS of the main accelerometer channel into a pandas object.

    The PVSS is scaled to units of mm/sec.
    """
    df_pvss = ch_data_cache._PVSSData
    if df_pvss.size == 0:
        return None

    df_pvss["Resultant"] = ch_data_cache._PVSSResultantData
    df_pvss = df_pvss * endaq.batch.analyzer.MPS_TO_MMPS

    return df_pvss.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


def _make_metrics(ch_data_cache):
    """
    Format the channel metrics of a recording into a pandas object.

    The following units listed by type are used for the metrics:
    - acceleration - g (gravity = 9.80665 meters per square second)
    - velocity - millimeters per second
    - displacement - millimeters
    - rotation speed - degrees per second
    - GPS position - degrees latitude/longitude
    - GPS speed - km/h (kilometers per hour)
    - audio - unitless
    - temperature - degrees Celsius
    - pressure - kiloPascals
    """
    df = pd.concat(
        [
            ch_data_cache.accRMSFull,
            ch_data_cache.velRMSFull,
            ch_data_cache.disRMSFull,
            ch_data_cache.accPeakFull,
            ch_data_cache.pseudoVelPeakFull,
            ch_data_cache.gpsLocFull,
            ch_data_cache.gpsSpeedFull,
            ch_data_cache.gyroRMSFull,
            ch_data_cache.micRMSFull,
            ch_data_cache.tempFull,
            ch_data_cache.pressFull,
        ],
        axis="columns",
    )

    # Format data into desired shape
    df.columns.name = "calculation"
    series = df.stack().reorder_levels(["calculation", "axis"])

    return series


def _make_peak_windows(ch_data_cache, margin_len):
    """
    Store windows of the main accelerometer channel about its peaks in a pandas
    object.

    The acceleration is scaled to units of g (gravity = 9.80665 meters per
    square second).
    """
    df_accel = ch_data_cache._accelerationData.copy()
    df_accel["Resultant"] = ch_data_cache._accelerationResultantData
    df_accel = endaq.batch.analyzer.MPS2_TO_G * df_accel
    if df_accel.size == 0:
        return None

    dt = endaq.calc.sample_spacing(df_accel)

    data_noidx = df_accel.reset_index(drop=True)
    peak_indices = data_noidx.abs().idxmax(axis="rows")
    aligned_peak_data = pd.concat(
        [
            pd.Series(
                df_accel[col].to_numpy(),
                index=(data_noidx.index - peak_indices[col]),
                name=col,
            )
            for col in df_accel.columns
        ],
        axis="columns",
    )
    aligned_peak_data = aligned_peak_data.loc[-margin_len : margin_len + 1]
    aligned_peak_data.index = pd.Series(
        aligned_peak_data.index * dt, name="peak offset"
    )
    aligned_peak_data.columns = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "axis": aligned_peak_data.columns.values,
                "peak time": df_accel.index[peak_indices],
            }
        )
    )

    # Format results
    result = (
        aligned_peak_data.stack()
        .stack()
        .reorder_levels(["axis", "peak time", "peak offset"])
    )

    return result


def _make_vc_curves(ch_data_cache):
    """
    Format the VC curves of the main accelerometer channel into a pandas object.
    """
    df_vc = (
        ch_data_cache._VCCurveData * endaq.batch.analyzer.MPS_TO_UMPS
    )  # (m/s) -> (μm/s)
    df_vc["Resultant"] = calc_stats.L2_norm(df_vc.to_numpy(), axis=1)
    if df_vc.size == 0:
        return None

    return df_vc.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


class GetDataBuilder:
    """
    The main interface for the calculations.

    This object has two types of functions:

    - *configuration functions* - these determine what calculations will be
      performed on IDE recordings, and pass in any requisite parameters for said
      calculations. This includes the following functions:

      - ``add_psd``
      - ``add_pvss``
      - ``add_metrics``
      - ``add_peaks``
      - ``add_vc_curves``

    - *execution functions* - these functions take recording files as parameters,
      perform the configured calculations on the data therein, and return the
      calculated data as a `OutputStruct` object that wraps pandas objects.

      This includes the functions ``_get_data`` & ``aggregate_data``, which
      operates on one & multiple file(s), respectively.

    A typical use case will look something like this:

    .. code-block:: python

        filenames = [...]

        calc_output = (
            GetDataBuilder(accel_highpass_cutoff=1)
            .add_psd(freq_bin_width=1)
            .add_pvss(init_freq=1, bins_per_octave=12)
            .add_metrics()
            .add_peaks(margin_len=100)
            .add_vc_curves(init_freq=1, bins_per_octave=3)
            .aggregate_data(filenames)
        )
        file_data = calc_output.dataframes
    """

    def __init__(
        self,
        *,
        preferred_chs=[],
        accel_highpass_cutoff,
        accel_start_time=None,
        accel_end_time=None,
        accel_start_margin=None,
        accel_end_margin=None,
        accel_integral_tukey_percent=0,
        accel_integral_zero="start",
    ):
        """
        :param preferred_chs: a sequence of channels; each channel listed is
            prioritized over others of the same type of physical measurement
            (e.g., acceleration, temperature, pressure, etc.)
        :param accel_highpass_cutoff: the cutoff frequency used when
            pre-filtering acceleration data
        :param accel_start_time: the relative timestamp before which to reject
            recording data; cannot be used in conjunction with
            `accel_start_margin`
        :param accel_end_time: the relative timestamp after which to reject
            recording data; cannot be used in conjunction with
            `accel_end_margin`
        :param accel_start_margin: the number of samples before which to reject
            recording data; cannot be used in conjunction with
            `accel_start_time`
        :param accel_end_margin: the numper of samples after which to reject
            recording data; cannot be used in conjunction with
            `accel_end_time`
        :param accel_integral_tukey_percent: the alpha parameter of a tukey
            window applied to the acceleration before integrating into
            velocity & displacement; see the `tukey_percent` parameter in
            ``endaq.calc.integrate.integrals`` for details
        :param accel_integral_zero: the output quantity driven to zero when
            integrating the acceleration into velocity & displacement; see the
            `zero` parameter in ``endaq.calc.integrate.integrals`` for details
        """
        if accel_start_time is not None and accel_start_margin is not None:
            raise ValueError(
                "only one of `accel_start_time` and `accel_start_margin` may be set at once"
            )
        if accel_end_time is not None and accel_end_margin is not None:
            raise ValueError(
                "only one of `accel_end_time` and `accel_end_margin` may be set at once"
            )

        self._metrics_queue = {}  # dict maintains insertion order, unlike set

        self._ch_data_cache_kwargs = dict(
            accel_highpass_cutoff=accel_highpass_cutoff,
            accel_start_time=accel_start_time,
            accel_end_time=accel_end_time,
            accel_start_margin=accel_start_margin,
            accel_end_margin=accel_end_margin,
            accel_integral_tukey_percent=accel_integral_tukey_percent,
            accel_integral_zero=accel_integral_zero,
        )
        self._preferred_chs = preferred_chs

        # Even unused parameters MUST be set; used to instantiate `CalcCache` in `_get_data`
        self._psd_freq_bin_width = None
        self._psd_freq_start_octave = None
        self._psd_bins_per_octave = None
        self._psd_window = None
        self._pvss_init_freq = None
        self._pvss_bins_per_octave = None
        self._peak_window_margin_len = None
        self._vc_init_freq = None
        self._vc_bins_per_octave = None

    def add_psd(
        self,
        *,
        freq_bin_width=None,
        freq_start_octave=None,
        bins_per_octave=None,
        window="hanning",
    ):
        """
        Add the acceleration PSD to the calculation queue.

        :param freq_bin_width: the desired spacing between adjacent PSD samples;
            a default is provided only if `bins_per_octave` is used, otherwise
            this parameter is required
        :param freq_start_octave: the first frequency to use in octave-spacing;
            this is only used if `bins_per_octave` is set
        :param bins_per_octave: the number of frequency bins per octave in a
            log-spaced PSD; if not set, the PSD will be linearly-spaced as
            specified by `freq_bin_width`
        :param window: the window type used in the PSD calculation; see the
            documentation for ``scipy.signal.welch`` for details
        """
        if all(i is None for i in (freq_bin_width, bins_per_octave)):
            raise ValueError(
                "must at least provide parameters for one of linear and log-spaced modes"
            )
        if freq_bin_width is None:
            if freq_start_octave is None:
                freq_start_octave = 1

            fstart_breadth = 2 ** (1 / (2 * bins_per_octave)) - 2 ** (
                -1 / (2 * bins_per_octave)
            )
            freq_bin_width = freq_start_octave / int(5 / fstart_breadth)

        self._metrics_queue["psd"] = None
        self._psd_freq_bin_width = freq_bin_width
        self._psd_freq_start_octave = freq_start_octave
        self._psd_bins_per_octave = bins_per_octave
        self._psd_window = window

        return self

    def add_pvss(self, *, init_freq, bins_per_octave):
        """
        Add the acceleration PVSS (Pseudo Velocity Shock Spectrum) to the
        calculation queue.

        :param init_freq: the first frequency sample in the spectrum
        :param bins_per_octave: the number of samples per frequency octave
        """
        self._metrics_queue["pvss"] = None
        self._pvss_init_freq = init_freq
        self._pvss_bins_per_octave = bins_per_octave

        return self

    def add_metrics(self):
        """Add broad channel metrics to the calculation queue."""
        self._metrics_queue["metrics"] = None

        if "pvss" not in self._metrics_queue:
            self._pvss_init_freq = 1
            self._pvss_bins_per_octave = 12

        return self

    def add_peaks(self, *, margin_len):
        """
        Add windows about the acceleration's peak value to the calculation
        queue.

        :param margin_len: the number of samples on each side of a peak to
            include in the windows
        """
        self._metrics_queue["peaks"] = None
        self._peak_window_margin_len = margin_len

        return self

    def add_vc_curves(self, *, init_freq, bins_per_octave):
        """
        Add Vibration Criteria (VC) Curves to the calculation queue.

        :param init_freq: the first frequency
        :param bins_per_octave:  the number of samples per frequency octave
        """
        self._metrics_queue["vc_curves"] = None

        if "psd" not in self._metrics_queue:
            self._psd_freq_bin_width = 0.2
            self._psd_window = "hanning"
        self._vc_init_freq = init_freq
        self._vc_bins_per_octave = bins_per_octave

        return self

    def _get_data(self, filename):
        """
        Calculate data from a single recording into a pandas object.

        Used internally by `aggregate_data`.
        """
        print(f"processing {filename}...")

        data = {}
        with endaq.ide.get_doc(filename) as ds:
            ch_data_cache = endaq.batch.analyzer.CalcCache.from_ide(
                ds,
                endaq.batch.analyzer.CalcParams(
                    **self._ch_data_cache_kwargs,
                    psd_window=self._psd_window,
                    psd_freq_bin_width=self._psd_freq_bin_width,
                    pvss_init_freq=self._pvss_init_freq,
                    pvss_bins_per_octave=self._pvss_bins_per_octave,
                    vc_init_freq=self._vc_init_freq,
                    vc_bins_per_octave=self._vc_bins_per_octave,
                ),
                preferred_chs=self._preferred_chs,
            )

            data["meta"] = _make_meta(ds)

            funcs = dict(
                psd=partial(
                    _make_psd,
                    fstart=self._psd_freq_start_octave,
                    bins_per_octave=self._psd_bins_per_octave,
                ),
                pvss=_make_pvss,
                metrics=_make_metrics,
                peaks=partial(
                    _make_peak_windows,
                    margin_len=self._peak_window_margin_len,
                ),
                vc_curves=_make_vc_curves,
            )
            for output_type in self._metrics_queue.keys():
                data[output_type] = funcs[output_type](ch_data_cache)

        return data

    def aggregate_data(self, filenames) -> OutputStruct:
        """
        Compile configured data from the given files into a dataframe.

        :param filenames: a sequence of paths of recording files to process
        """
        series_lists = zip(
            *(self._get_data(filename).values() for filename in filenames)
        )

        print("aggregating data...")
        meta, *dfs = (
            pd.concat(
                series_list,
                keys=filenames,
                names=["filename"]
                + next(s for s in series_list if s is not None).index.names,
            )
            if series_list and any(s is not None for s in series_list)
            else None
            for series_list in series_lists
        )

        meta = meta.unstack(level=1)

        def reformat(series):
            if series is None:
                return None

            df = series.to_frame().T.melt()
            df["serial number"] = meta.loc[df["filename"], "serial number"].reset_index(
                drop=True
            )
            df["start time"] = meta.loc[df["filename"], "start time"].reset_index(
                drop=True
            )

            return df

        dfs = dict(
            meta=meta,
            **{key: reformat(df) for (key, df) in zip(self._metrics_queue.keys(), dfs)},
        )

        print("done!")

        return OutputStruct(dfs)


class OutputStruct:
    """
    A data wrapper class with methods for common export operations.

    Objects of this class are generated by ``GetDataBuilder.aggregate_data``.
    This class is not intended be instantiated manually.
    """

    def __init__(self, data):
        self.dataframes: List[pd.DataFrame] = data

    def to_csv_folder(self, folder_path):
        """
        Write data to a folder as CSV's.

        :param folder_path: the output directory path for .CSV files
        """
        os.makedirs(folder_path, exist_ok=True)

        for k, df in self.dataframes.items():
            path = os.path.join(folder_path, f"{k}.csv")
            df.to_csv(path, index=(k == "meta"))

    def to_html_plots(self, folder_path=None, show=False):
        """
        Generate plots in HTML.

        :param folder_path: the output directory for saving .HTML
            plots; if `None` (default), plots are not saved
        :param show: whether to open plots after generation; defaults to `False`
        """
        if not any((folder_path, show)):
            return

        import plotly.express as px

        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        for k, df in self.dataframes.items():
            if k == "meta":
                continue
            if k == "psd":
                fig = px.line(
                    df,
                    x="frequency (Hz)",
                    y="value",
                    color="filename",
                    line_dash="axis",
                )

                fig.update_xaxes(type="log", title_text="frequency (Hz)")
                fig.update_yaxes(type="log", title_text="Acceleration (g^2/Hz)")
                fig.update_layout(title="Acceleration PSD")

            elif k == "pvss":
                fig = px.line(
                    df,
                    x="frequency (Hz)",
                    y="value",
                    color="filename",
                    line_dash="axis",
                )
                fig.update_xaxes(type="log", title_text="frequency (Hz)")
                fig.update_yaxes(type="log", title_text="Velocity (mm/s)")
                fig.update_layout(title="Pseudo Velocity Shock Spectrum (PVSS)")

            elif k == "metrics":
                warnings.warn("HTML plot for metrics not currently implemented")
                continue

            elif k == "peaks":
                fig = px.line(
                    df,
                    x=df["peak offset"].dt.total_seconds(),
                    # ^ plotly doesn't handle timedelta's well
                    y="value",
                    color="filename",
                    line_dash="axis",
                )
                fig.update_xaxes(title_text="time relative to peak (s)")
                fig.update_yaxes(title_text="Acceleration (g)")
                fig.update_layout(title="Window about Acceleration Peaks")

            elif k == "vc_curves":
                fig = px.line(
                    df,
                    x="frequency (Hz)",
                    y="value",
                    color="filename",
                    line_dash="axis",
                )

                fig.update_xaxes(type="log", title_text="frequency (Hz)")
                fig.update_yaxes(
                    type="log", title_text="1/3-Octave RMS Velocity (μm/s)"
                )
                fig.update_layout(title="Vibration Criteria (VC) Curves")

            else:
                raise RuntimeError(f"no configuration for plotting '{k}' data")

            if not folder_path and show:
                fig.show()
            else:
                fig.write_html(
                    file=os.path.join(folder_path, f"{k}.html"),
                    include_plotlyjs="directory",
                    full_html=True,
                    auto_open=show,
                )
