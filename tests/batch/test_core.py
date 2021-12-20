from collections import namedtuple
import os
import tempfile

import idelib
import numpy as np
import pandas as pd
import pytest

import endaq.batch.core
import endaq.batch.analyzer
from endaq.batch.utils import ide_utils


@pytest.mark.parametrize(
    "filename, expt_result",
    [
        (
            os.path.join("tests", "batch", "test1.IDE"),
            [10118, np.datetime64("2020-09-16 19:05:49.771728")],
        ),
        (
            os.path.join("tests", "batch", "test2.IDE"),
            [10118, np.datetime64("2020-09-16 19:04:22.475738")],
        ),
        (
            os.path.join("tests", "batch", "test4.IDE"),
            [10118, np.datetime64("2020-11-18 17:31:27.000000")],
        ),
    ],
)
def test_make_meta(filename, expt_result):
    with idelib.importFile(filename) as ds:
        meta = endaq.batch.core._make_meta(ds)

    assert np.all(meta.index == ["serial number", "start time"])
    assert np.all(meta == expt_result)


@pytest.mark.parametrize(
    "filename",
    [
        os.path.join("tests", "batch", "SSX70065.IDE"),
        os.path.join("tests", "batch", "test1.IDE"),
        os.path.join("tests", "batch", "test2.IDE"),
    ],
)
def test_make_peak_windows(filename):
    with idelib.importFile(filename) as ds:
        accel_ch = ide_utils.get_ch_type_best(ds, "acc")

        data = accel_ch.to_pandas(time_mode="timedelta")
        utc_start_time = ds.lastUtcTime
        axis_names = accel_ch.axis_names

        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ds,
            endaq.batch.analyzer.CalcParams(
                accel_highpass_cutoff=None,
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=None,
                psd_window="hann",
                pvss_init_freq=None,
                pvss_bins_per_octave=None,
                vc_init_freq=None,
                vc_bins_per_octave=None,
            ),
        )
        calc_meta = endaq.batch.core._make_meta(ds)
        calc_peaks = endaq.batch.core._make_peak_windows(analyzer, margin_len=10)
        i_max = np.argmax(np.abs(analyzer._accelerationData.to_numpy()), axis=0)

    assert calc_peaks.index.names == ["axis", "peak time", "peak offset"]
    assert np.all(
        calc_peaks.index.unique(level="axis").sort_values()
        == ["Resultant"] + axis_names
    )

    calc_peak_times = (
        calc_peaks.index.droplevel("peak offset")
        .unique()
        .to_frame()
        .droplevel("peak time")
        .loc[axis_names, "peak time"]
    )
    expt_peak_times = data.index[i_max].astype("timedelta64[ns]")
    assert np.all(calc_peak_times == expt_peak_times)


@pytest.fixture
def data_builder():
    return (
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1)
        .add_psd(freq_bin_width=1)
        .add_pvss(init_freq=1, bins_per_octave=12)
        .add_pvss_halfsine_envelope()
        .add_metrics()
        .add_peaks(margin_len=1000)
        .add_vc_curves(init_freq=1, bins_per_octave=3)
    )


@pytest.mark.parametrize(
    "filename",
    [
        os.path.join("tests", "batch", "SSX70065.IDE"),
        os.path.join("tests", "batch", "test1.IDE"),
        os.path.join("tests", "batch", "test2.IDE"),
        os.path.join("tests", "batch", "test3.IDE"),
        os.path.join("tests", "batch", "test4.IDE"),
        os.path.join("tests", "batch", "test5.IDE"),
        os.path.join("tests", "batch", "GPS-Chick-Fil-A_003.IDE"),
        os.path.join("tests", "batch", "High-Drop.IDE"),
    ],
)
@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:no acceleration channel in:UserWarning")
def test_get_data(filename):
    """Test `_get_data` over several varieties of recording files."""
    (
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1)
        .add_psd(freq_bin_width=1)
        .add_pvss(init_freq=1, bins_per_octave=12)
        .add_pvss_halfsine_envelope()
        .add_metrics()
        .add_peaks(margin_len=1000)
        .add_vc_curves(init_freq=1, bins_per_octave=3)
        ._get_data(filename)
    )


def assert_output_is_valid(output: endaq.batch.core.OutputStruct):
    """Validate the contents & structure of an `OutputStruct` object."""
    assert isinstance(output, endaq.batch.core.OutputStruct)
    assert isinstance(output.dataframes, dict)
    assert {
        "meta",
        "psd",
        "pvss",
        "halfsine",
        "metrics",
        "peaks",
        "vc_curves",
    }.issuperset(output.dataframes)

    assert output.dataframes["meta"].index.name == "filename"
    assert output.dataframes["meta"].columns.to_list() == [
        "serial number",
        "start time",
    ]

    if "psd" in output.dataframes:
        assert np.all(
            output.dataframes["psd"].columns
            == [
                "filename",
                "axis",
                "frequency (Hz)",
                "value",
                "serial number",
                "start time",
            ]
        )

    if "pvss" in output.dataframes:
        assert np.all(
            output.dataframes["pvss"].columns
            == [
                "filename",
                "axis",
                "frequency (Hz)",
                "value",
                "serial number",
                "start time",
            ]
        )

    if "halfsine" in output.dataframes:
        assert np.all(
            output.dataframes["halfsine"].columns
            == [
                "filename",
                "axis",
                "timestamp",
                "value",
                "serial number",
                "start time",
            ]
        )

    if "metrics" in output.dataframes:
        assert np.all(
            output.dataframes["metrics"].columns
            == [
                "filename",
                "calculation",
                "axis",
                "value",
                "serial number",
                "start time",
            ]
        )

    if "peaks" in output.dataframes:
        assert np.all(
            output.dataframes["peaks"].columns
            == [
                "filename",
                "axis",
                "peak time",
                "peak offset",
                "value",
                "serial number",
                "start time",
            ]
        )

    if "vc_curves" in output.dataframes:
        assert np.all(
            output.dataframes["vc_curves"].columns
            == [
                "filename",
                "axis",
                "frequency (Hz)",
                "value",
                "serial number",
                "start time",
            ]
        )


@pytest.mark.parametrize(
    "getdata_builder",
    [
        # Each builder piece individually
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_psd(
            freq_bin_width=1
        ),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1)
        .add_pvss(init_freq=1, bins_per_octave=12)
        .add_pvss_halfsine_envelope(),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_metrics(),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_peaks(
            margin_len=1000
        ),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_vc_curves(
            init_freq=1, bins_per_octave=3
        ),
        # All builder pieces altogether
        (
            endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1)
            .add_psd(freq_bin_width=1)
            .add_pvss(init_freq=1, bins_per_octave=12)
            .add_pvss_halfsine_envelope()
            .add_metrics()
            .add_peaks(margin_len=1000)
            .add_vc_curves(init_freq=1, bins_per_octave=3)
        ),
        # Disable highpass filter
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=None).add_psd(
            freq_bin_width=1
        ),
        # Test time restrictions on acceleration data
        endaq.batch.core.GetDataBuilder(
            accel_highpass_cutoff=1,
            accel_start_time=np.timedelta64(5, "s"),
            accel_end_time=np.timedelta64(10, "s"),
        ).add_psd(freq_bin_width=1),
        endaq.batch.core.GetDataBuilder(
            accel_highpass_cutoff=1,
            accel_start_margin=np.timedelta64(2, "s"),
            accel_end_margin=np.timedelta64(2, "s"),
        ).add_psd(freq_bin_width=1),
        endaq.batch.core.GetDataBuilder(
            accel_highpass_cutoff=1,
            accel_start_time=np.timedelta64(5, "s"),
            accel_end_margin=np.timedelta64(2, "s"),
        ).add_psd(freq_bin_width=1),
        endaq.batch.core.GetDataBuilder(
            accel_highpass_cutoff=1,
            accel_start_margin=np.timedelta64(2, "s"),
            accel_end_time=np.timedelta64(10, "s"),
        ).add_psd(freq_bin_width=1),
        # Octave-spaced PSD parameters
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_psd(
            bins_per_octave=1
        ),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_psd(
            freq_start_octave=0.1, bins_per_octave=12
        ),
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1).add_psd(
            freq_bin_width=0.2, bins_per_octave=3
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:no acceleration channel in:UserWarning")
@pytest.mark.filterwarnings(
    "ignore"
    ":nperseg .* is greater than input length .*, using nperseg .*"
    ":UserWarning"
)
def test_aggregate_data(getdata_builder):
    """Test `aggregate_data` over several configurations of `GetDataBuilder`."""
    filenames = [
        os.path.join("tests", "batch", "test1.IDE"),
        os.path.join("tests", "batch", "test2.IDE"),
        os.path.join("tests", "batch", "test4.IDE"),
    ]

    calc_result = getdata_builder.aggregate_data(filenames)

    assert list(calc_result.dataframes)[1:] == list(getdata_builder._metrics_queue)
    assert len(calc_result.dataframes["meta"]) == 3
    assert_output_is_valid(calc_result)


@pytest.fixture
def output_struct():
    data = {}

    fieldname_mods = dict(
        frequency="frequency (Hz)",
        serial_number="serial number",
        start_time="start time",
        peak_time="peak time",
        peak_offset="peak offset",
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "serial_number",
            "start_time",
        ],
    )
    data["meta"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        index="filename",
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "axis",
            "frequency",
            "value",
            "serial_number",
            "start_time",
        ],
    )
    data["psd"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=1.0,
                value=10,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=2.0,
                value=5,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=1.0,
                value=8,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=2.0,
                value=16,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "axis",
            "frequency",
            "value",
            "serial_number",
            "start_time",
        ],
    )
    data["pvss"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=1.0,
                value=100,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=2.0,
                value=50,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=1.0,
                value=80,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=2.0,
                value=160,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "calculation",
            "axis",
            "value",
            "serial_number",
            "start_time",
        ],
    )
    data["metrics"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                calculation="RMS Acceleration",
                axis="X",
                value=0.2,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                calculation="RMS Acceleration",
                axis="Y",
                value=0.1,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "axis",
            "peak_time",
            "peak_offset",
            "value",
            "serial_number",
            "start_time",
        ],
    )
    data["peaks"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                axis="X",
                peak_time=np.timedelta64(3, "s"),
                peak_offset=np.timedelta64(-100, "us"),
                value=0.7,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub1.ide",
                axis="X",
                peak_time=np.timedelta64(3, "s"),
                peak_offset=np.timedelta64(0, "us"),
                value=1.1,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub1.ide",
                axis="X",
                peak_time=np.timedelta64(3, "s"),
                peak_offset=np.timedelta64(100, "us"),
                value=0.4,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                peak_time=np.timedelta64(5, "s"),
                peak_offset=np.timedelta64(-100, "us"),
                value=-1.2,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                peak_time=np.timedelta64(5, "s"),
                peak_offset=np.timedelta64(0, "us"),
                value=-1.5,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                peak_time=np.timedelta64(5, "s"),
                peak_offset=np.timedelta64(100, "us"),
                value=-0.9,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    RowStruct = namedtuple(
        "RowStruct",
        [
            "filename",
            "axis",
            "frequency",
            "value",
            "serial_number",
            "start_time",
        ],
    )
    data["vc_curves"] = pd.DataFrame.from_records(
        [
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=1.0,
                value=1000,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub1.ide",
                axis="X",
                frequency=2.0,
                value=500,
                serial_number=12345,
                start_time=np.datetime64("2020-01-01 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=1.0,
                value=800,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
            RowStruct(
                filename="stub2.ide",
                axis="Y",
                frequency=2.0,
                value=1600,
                serial_number=67890,
                start_time=np.datetime64("2020-02-02 00:00:00"),
            ),
        ],
        columns=[fieldname_mods.get(i, i) for i in RowStruct._fields],
    )

    result = endaq.batch.core.OutputStruct(data)
    assert_output_is_valid(result)

    return result


def test_output_to_csv_folder(output_struct):
    with tempfile.TemporaryDirectory() as dirpath:
        output_struct.to_csv_folder(dirpath)

        for k, v in output_struct.dataframes.items():
            filepath = os.path.join(dirpath, k + ".csv")
            assert os.path.isfile(filepath)

            read_result = pd.read_csv(
                filepath,
                **(dict(index_col="filename") if k == "meta" else {}),
            )
            assert v.astype(str).compare(read_result.astype(str)).size == 0


@pytest.mark.filterwarnings(
    "ignore:HTML plot for metrics not currently implemented:UserWarning"
)
def test_output_to_html_plots(output_struct):
    with tempfile.TemporaryDirectory() as dirpath:
        output_struct.to_html_plots(folder_path=dirpath, show=False)

        for k in output_struct.dataframes:
            # Not all dataframes get plotted
            if k in ("meta", "metrics"):
                continue

            filepath = os.path.join(dirpath, k + ".html")
            assert os.path.isfile(filepath)

            # can't do much else for validation...
