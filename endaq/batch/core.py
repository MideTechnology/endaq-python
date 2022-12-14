from __future__ import annotations

import typing
from typing import Any, Dict, List, Callable, Optional, Iterable

from dataclasses import dataclass
from functools import partial
import warnings
import os

import numpy as np
import pandas as pd

import endaq.ide
from endaq.calc import stats as calc_stats
from endaq.calc import psd as calc_psd

from endaq.batch import analyzer


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


def _make_psd(ch_data_cache: analyzer.CalcCache, fstart=None, bins_per_octave=None):
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
    df_psd = df_psd * analyzer.MPS2_TO_G ** 2  # (m/s^2)^2/Hz -> g^2/Hz

    return df_psd.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


def _make_pvss(ch_data_cache: analyzer.CalcCache):
    """
    Format the PVSS of the main accelerometer channel into a pandas object.

    The PVSS is scaled to units of mm/sec.
    """
    df_pvss = ch_data_cache._PVSSData
    if df_pvss.size == 0:
        return None

    df_pvss["Resultant"] = ch_data_cache._PVSSResultantData
    df_pvss = df_pvss * analyzer.MPS_TO_MMPS

    return df_pvss.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


def _make_halfsine_pvss_envelope(ch_data_cache, *args, **kwargs):
    df_pvss = ch_data_cache._PVSSData.copy()
    df_pvss["Resultant"] = ch_data_cache._PVSSResultantData
    df_pvss = df_pvss * analyzer.MPS_TO_MMPS
    if df_pvss.size == 0:
        return None

    return (
        endaq.calc.shock.enveloping_half_sine(df_pvss)
        .to_time_series(*args, **kwargs)
        .stack(level="axis")
        .reorder_levels(["axis", "timestamp"])
    )


def _make_metrics(
    ch_data_cache: analyzer.CalcCache,
    include: Iterable[str] = [],
    exclude: Iterable[str] = [],
):
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
    if include and exclude:
        raise ValueError("parameters `include` and `exclude` are mutually-exclusive")

    DEFAULT_EXCLUDE = [x.casefold() for x in ["RMS Sound Pressure"]]
    VALID_METRICS = {
        k.casefold(): v
        for (k, v) in [
            ("RMS Acceleration", "accRMSFull"),
            ("RMS Velocity", "velRMSFull"),
            ("RMS Displacement", "disRMSFull"),
            ("Peak Absolute Acceleration", "accPeakFull"),
            ("Peak Pseudo Velocity Shock Spectrum", "pseudoVelPeakFull"),
            ("GPS Position", "gpsLocFull"),
            ("GPS Speed", "gpsSpeedFull"),
            ("RMS Angular Velocity", "gyroRMSFull"),
            ("RMS Sound Pressure", "micRMSFull"),
            ("Average Sound Pressure Level", "micDecibelsFull"),
            ("Average Temperature", "tempFull"),
            ("Average Pressure", "pressFull"),
            ("Average Relative Humidity", "humidFull"),
        ]
    }
    assert sorted(VALID_METRICS.values()) == sorted(
        attr
        for attr in dir(analyzer.CalcCache)
        if not attr.startswith("_") and attr.endswith("Full")
    )

    include = [x.casefold() for x in include]
    exclude = [x.casefold() for x in exclude]

    invalid_metrics = set(include or exclude) - set(VALID_METRICS)
    if invalid_metrics:
        raise ValueError(f"invalid metrics {list(invalid_metrics)}")

    metric_names: Iterable[str]
    if include:
        metric_names = include
    elif exclude:
        exclude = set(exclude)
        metric_names = (x for x in VALID_METRICS if x not in exclude)
    else:
        metric_names = (x for x in VALID_METRICS if x not in DEFAULT_EXCLUDE)
    metric_attrs = [VALID_METRICS[name] for name in metric_names]

    df = pd.concat(
        [getattr(ch_data_cache, attr) for attr in metric_attrs],
        axis="columns",
    )

    # Format data into desired shape
    df.columns.name = "calculation"
    series = df.stack().reorder_levels(["calculation", "axis"])

    return series


def _make_peak_windows(ch_data_cache: analyzer.CalcCache, margin_len):
    """
    Store windows of the main accelerometer channel about its peaks in a pandas
    object.

    The acceleration is scaled to units of g (gravity = 9.80665 meters per
    square second).
    """
    df_accel = ch_data_cache._accelerationData.copy()
    df_accel["Resultant"] = ch_data_cache._accelerationResultantData
    df_accel = analyzer.MPS2_TO_G * df_accel
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


def _make_vc_curves(ch_data_cache: analyzer.CalcCache):
    """
    Format the VC curves of the main accelerometer channel into a pandas object.
    """
    df_vc = ch_data_cache._VCCurveData * analyzer.MPS_TO_UMPS  # (m/s) -> (μm/s)
    df_vc["Resultant"] = calc_stats.L2_norm(df_vc.to_numpy(), axis=1)
    if df_vc.size == 0:
        return None

    return df_vc.stack(level="axis").reorder_levels(["axis", "frequency (Hz)"])


class GetDataBuilder:
    """
    The main interface for calculations in ``endaq.batch``.

    This object has two types of functions:

    - *configuration functions* - these determine what calculations will be
      performed on IDE recordings, and pass in any requisite parameters for said
      calculations. This includes the following functions:

      .. hlist::

          - :py:meth:`add_psd`
          - :py:meth:`add_pvss`
          - :py:meth:`add_pvss_halfsine_envelope`
          - :py:meth:`add_metrics`
          - :py:meth:`add_peaks`
          - :py:meth:`add_vc_curves`

    - *execution functions* - these functions take recording files as parameters,
      perform the configured calculations on the data therein, and return the
      calculated data as a :py:class:`.OutputStruct` object that wraps pandas
      objects.

      This includes the functions :py:meth:`_get_data` &
      :py:meth:`aggregate_data`, which operates on one & multiple file(s),
      respectively.

    A typical use case will look something like this:

    .. code-block:: python

        filenames = [...]

        calc_output = (
            GetDataBuilder(accel_highpass_cutoff=1)
            .add_psd(freq_bin_width=1)
            .add_pvss(init_freq=1, bins_per_octave=12)
            .add_pvss_halfsine_envelope()
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
        :param accel_end_margin: the number of samples after which to reject
            recording data; cannot be used in conjunction with
            `accel_end_time`
        :param accel_integral_tukey_percent: the alpha parameter of a Tukey
            window applied to the acceleration before integrating into
            velocity & displacement; see the `tukey_percent` parameter in
            :py:func:`endaq.calc.integrate.integrals` for details
        :param accel_integral_zero: the output quantity driven to zero when
            integrating the acceleration into velocity & displacement; see the
            `zero` parameter in :py:func:`endaq.calc.integrate.integrals` for
            details
        """
        if accel_start_time is not None and accel_start_margin is not None:
            raise ValueError(
                "only one of `accel_start_time` and `accel_start_margin` may be set at once"
            )
        if accel_end_time is not None and accel_end_margin is not None:
            raise ValueError(
                "only one of `accel_end_time` and `accel_end_margin` may be set at once"
            )

        self._metrics_queue: List[Tuple[str, Callable[[analyzer.CalcCache], Any]]] = []

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
        self._psd_freq_bin_width_oct = None
        self._psd_window = None
        self._pvss_init_freq = None
        self._pvss_bins_per_octave = None
        self._vc_init_freq = None
        self._vc_bins_per_octave = None

    def add_psd(
        self,
        freq_bin_width: Optional[float] = None,
        freq_start_octave: Optional[float] = None,
        bins_per_octave: Optional[float] = None,
        window: Optional[str] = None,
    ):
        """
        Add the acceleration PSD to the calculation queue.

        *calculation output units*: :math:`\\frac{\\text{G}^2}{\\text{Hz}}`,
        where `G` is the acceleration of gravity :math:`\\left( 1 \\text{G}
        \\approx 9.80665 \\frac{ \\text{m} }{ \\text{sec}^2 } \\right)`

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

            freq_bin_width = endaq.calc.psd._aligned_bin_width(
                freq_start_octave, bins_per_octave
            )
            self._psd_freq_bin_width_oct = min(
                freq_bin_width, self._psd_freq_bin_width_oct or float("inf")
            )
        else:
            self._psd_freq_bin_width = freq_bin_width

        if not (self._psd_window is None or self._psd_window == window):
            raise ValueError(
                "inconsistent PSD windows provided:"
                f" first {self._psd_window}, then {window}"
            )
        self._psd_window = window

        self._metrics_queue.append(
            (
                "psd",
                partial(
                    _make_psd,
                    fstart=freq_start_octave,
                    bins_per_octave=bins_per_octave,
                ),
            )
        )

        return self

    def add_pvss(self, init_freq: float = 1.0, bins_per_octave: float = 3.0):
        """
        Add the acceleration PVSS (Pseudo Velocity Shock Spectrum) to the
        calculation queue.

        *calculation output units*: :math:`\\frac{\\text{mm}}{\\text{sec}}`

        :param init_freq: the first frequency sample in the spectrum
        :param bins_per_octave: the number of samples per frequency octave
        """
        if any(name == "pvss" for (name, _) in self._metrics_queue):
            raise RuntimeError('cannot call "add_pvss" twice')

        self._metrics_queue.append(("pvss", _make_pvss))
        self._pvss_init_freq = init_freq
        self._pvss_bins_per_octave = bins_per_octave

        return self

    def add_pvss_halfsine_envelope(
        self,
        tstart: Optional[float] = None,
        tstop: Optional[float] = None,
        dt: Optional[float] = None,
        tpulse: Optional[float] = None,
    ):
        """
        Add the half-sine envelope for the acceleration's PVSS (Pseudo Velocity
        Shock Spectrum) to the calculation queue.

        *calculation output units*: :math:`\\frac{\\text{mm}}{\\text{sec}}`
        """
        self._metrics_queue.append(
            (
                "halfsine",
                partial(
                    _make_halfsine_pvss_envelope,
                    tstart=tstart,
                    tstop=tstop,
                    dt=dt,
                    tpulse=tpulse,
                ),
            )
        )

        return self

    def add_metrics(self, include: List[str] = [], exclude: List[str] = []):
        """
        Add broad channel metrics to the calculation queue.

        The output units for each metric are listed below:

        .. hlist::

            - `RMS Acceleration`: :math:`\\text{G}`
            - `RMS Velocity`: :math:`\\frac{\\text{mm}}{\\text{sec}}`
            - `RMS Displacement`: :math:`\\text{mm}`
            - `Peak Absolute Acceleration`: :math:`\\text{G}`
            - `Peak Pseudo Velocity Shock Spectrum`: :math:`\\frac{\\text{mm}}{\\text{sec}}`
            - `GPS Position`: :math:`\\text{degrees}`
            - `GPS Speed`: :math:`\\frac{\\text{km}}{\\text{hr}}`
            - `RMS Angular Velocity`: :math:`\\frac{\\text{degrees}}{\\text{sec}}`
            - `RMS Microphone`: :math:`\\text{Pascals}`
            - `Average Temperature`: :math:`{}^{\\circ} \\text{C}`
            - `Average Pressure`: :math:`\\text{Pascals}`
            - `Average Relative Humidity`: :math:`\\text{%}`

        where `G` is the acceleration of gravity :math:`\\left( 1 \\text{G}
        \\approx 9.80665 \\frac{ \\text{m} }{ \\text{sec}^2 } \\right)`

        """
        self._metrics_queue.append(
            ("metrics", partial(_make_metrics, include=include, exclude=exclude))
        )

        # no PSD metrics -> no need to provide PSD params

        # Need to provide default PVSS metrics
        if self._pvss_init_freq is None:
            self._pvss_init_freq = 1
            self._pvss_bins_per_octave = 12

        return self

    def add_peaks(self, margin_len: int = 1000):
        """
        Add windows about the acceleration's peak value to the calculation
        queue.

        *calculation output units*: :math:`\\text{G}`, where `G` is the
        acceleration of gravity :math:`\\left( 1 \\text{G} \\approx 9.80665
        \\frac{ \\text{m} }{ \\text{sec}^2 } \\right)`

        :param margin_len: the number of samples on each side of a peak to
            include in the windows
        """
        self._metrics_queue.append(
            (
                "peaks",
                partial(
                    _make_peak_windows,
                    margin_len=margin_len,
                ),
            )
        )

        return self

    def add_vc_curves(self, init_freq: float = 1.0, bins_per_octave: float = 3.0):
        """
        Add Vibration Criteria (VC) Curves to the calculation queue.

        *calculation output units*: :math:`\\frac{\\text{μm}}{\\text{sec}}`

        :param init_freq: the first frequency
        :param bins_per_octave:  the number of samples per frequency octave
        """
        self._metrics_queue.append(("vc_curves", _make_vc_curves))

        if "psd" not in self._metrics_queue:
            self._psd_freq_bin_width_oct = min(
                0.2,  # TODO: use `endaq.calc.psd._aligned_bin_width`
                self._psd_freq_bin_width_oct or float("inf"),
            )
        self._vc_init_freq = init_freq
        self._vc_bins_per_octave = bins_per_octave

        return self

    def _make_calc_params(self) -> analyzer.CalcParams:
        return analyzer.CalcParams(
            **self._ch_data_cache_kwargs,
            psd_window=self._psd_window or "hann",
            psd_freq_bin_width=self._psd_freq_bin_width or self._psd_freq_bin_width_oct,
            pvss_init_freq=self._pvss_init_freq,
            pvss_bins_per_octave=self._pvss_bins_per_octave,
            vc_init_freq=self._vc_init_freq,
            vc_bins_per_octave=self._vc_bins_per_octave,
        )

    def _get_data(self, filename):
        """
        Calculate data from a single recording into a pandas object.

        Used internally by `aggregate_data`.
        """
        print(f"processing {filename}...")

        data = []
        with endaq.ide.get_doc(filename) as ds:
            ch_data_cache = analyzer.CalcCache.from_ide(
                ds,
                self._make_calc_params(),
                preferred_chs=self._preferred_chs,
            )

            data.append(("meta", _make_meta(ds)))

            for output_type, func in self._metrics_queue:
                data.append((output_type, func(ch_data_cache)))

        return data

    def aggregate_data(self, filenames) -> Optional[OutputStruct]:
        """
        Compile configured data from the given files into a dataframe.

        :param filenames: a sequence of paths of recording files to process
        """
        if len(filenames) == 0:
            return None

        http_files, local_files = [], []
        for file in filenames:
            path_formatted, mode = endaq.ide.files.normalized_path(file)

            if mode == "url":
                http_files.append(file)
            else:  # mode == "local"
                local_files.append(path_formatted)

        if len(local_files) == 0:
            root_path = ""
        elif len(local_files) == 1:
            # Common path will take the one file's whole path as the "root path"
            # -> remove the basename from this path
            root_path = os.path.dirname(local_files[0])
        else:
            root_path = os.path.commonpath(local_files)

        files = http_files + local_files
        display_names = http_files + [
            os.path.relpath(name, start=root_path) for name in local_files
        ]

        series_lists = zip(*([d for (_k, d) in self._get_data(file)] for file in files))

        print("aggregating data...")
        meta, *dfs = (
            pd.concat(
                series_list,
                keys=display_names,
                names=["filename"]
                + next(s for s in series_list if s is not None).index.names,
            )
            if series_list and any(s is not None for s in series_list)
            else None
            for series_list in series_lists
        )

        meta = meta.unstack(level=1)
        meta.attrs["rootpath"] = root_path

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

        dfs = [("meta", meta)] + [
            (df_type, reformat(df))
            for ((df_type, _), df) in zip(self._metrics_queue, dfs)
        ]

        print("done!")

        return OutputStruct(dfs)


class OutputStruct:
    """
    A data wrapper class with methods for common export operations.

    Objects of this class are generated by :py:meth:`.GetDataBuilder.aggregate_data`.
    This class is not intended be instantiated manually.
    """

    def __init__(self, data: List[Tuple[str, pd.DataFrame]]):
        self.dataframes = data

    def to_csv_folder(self, folder_path):
        """
        Write data to a folder as CSV's.

        :param folder_path: the output directory path for .CSV files
        """
        os.makedirs(folder_path, exist_ok=True)

        for k, df in self.dataframes:
            path = os.path.join(folder_path, f"{k}.csv")
            df.to_csv(path, index=(k == "meta"))

    def to_html_plots(
        self,
        folder_path=None,
        show: bool = False,
        theme: typing.Literal[
            None, "endaq", "endaq_light", "endaq_arial", "endaq_light_arial"
        ] = "endaq",
    ):
        """
        Generate plots in HTML.

        :param folder_path: The output directory for saving .HTML
            plots. If `None` (default), plots are not saved.
        :param show: Whether to open plots after generation. Defaults to `False`.
        :param theme: The enDAQ plotly theme to use; see
            :py:func:`endaq.plot.utilities.set_theme` for details on the
            supported options. Defaults to `"endaq"`. If `None`, the default
            Plotly theme is used.
        """
        if not any((folder_path, show)):
            return

        import plotly.express as px

        if theme is not None:
            from endaq.plot.utilities import set_theme

            set_theme(theme)

        if folder_path:
            os.makedirs(folder_path, exist_ok=True)

        for k, df in self.dataframes:
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

            elif k == "halfsine":
                fig = px.line(
                    df,
                    x="timestamp",
                    y="value",
                    color="filename",
                    line_dash="axis",
                )
                fig.update_layout(title="PVSS Enveloping Half Sine Pulses")

            elif k == "metrics":
                warnings.warn("HTML plot for metrics not currently implemented")
                continue

            elif k == "peaks":
                fig = px.line(
                    df,
                    x=df["peak offset"],
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
