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
            os.path.join(os.path.dirname(__file__), "test1.IDE"),
            [10118, np.datetime64("2020-09-16 19:05:49.771728")],
        ),
        (
            os.path.join(os.path.dirname(__file__), "test2.IDE"),
            [10118, np.datetime64("2020-09-16 19:04:22.475738")],
        ),
        (
            os.path.join(os.path.dirname(__file__), "test4.IDE"),
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
        os.path.join(os.path.dirname(__file__), "SSX70065.IDE"),
        os.path.join(os.path.dirname(__file__), "test1.IDE"),
        os.path.join(os.path.dirname(__file__), "test2.IDE"),
    ],
)
def test_make_peak_windows(filename):
    with idelib.importFile(filename) as ds:
        accel_ch = ide_utils.get_ch_type_best(ds, "Acceleration")

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


@pytest.mark.parametrize(
    "filename",
    [
        os.path.join(os.path.dirname(__file__), "SSX70065.IDE"),
        os.path.join(os.path.dirname(__file__), "test1.IDE"),
        os.path.join(os.path.dirname(__file__), "test3.IDE"),
        os.path.join(os.path.dirname(__file__), "test5.IDE"),
        os.path.join(os.path.dirname(__file__), "GPS-Chick-Fil-A_003.IDE"),
        "https://info.endaq.com/hubfs/data/High-Drop.ide",
        "https://info.endaq.com/hubfs/data/Punching-Bag.ide",
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
    assert isinstance(output.dataframes, list)
    assert {
        "meta",
        "psd",
        "pvss",
        "halfsine",
        "metrics",
        "peaks",
        "vc_curves",
    }.issuperset({k for (k, v) in output.dataframes})
    assert any(name == "meta" for (name, _) in output.dataframes)

    for (name, df) in output.dataframes:
        if name == "meta":
            assert df.index.name == "filename"
            assert df.columns.to_list() == [
                "serial number",
                "start time",
            ]

        if name == "psd":
            assert np.all(
                df.columns
                == [
                    "filename",
                    "axis",
                    "frequency (Hz)",
                    "value",
                    "serial number",
                    "start time",
                ]
            )

        if name == "pvss":
            assert np.all(
                df.columns
                == [
                    "filename",
                    "axis",
                    "frequency (Hz)",
                    "value",
                    "serial number",
                    "start time",
                ]
            )

        if name == "halfsine":
            assert np.all(
                df.columns
                == [
                    "filename",
                    "axis",
                    "timestamp",
                    "value",
                    "serial number",
                    "start time",
                ]
            )

        if name == "metrics":
            assert np.all(
                df.columns
                == [
                    "filename",
                    "calculation",
                    "axis",
                    "value",
                    "serial number",
                    "start time",
                ]
            )

        if name == "peaks":
            assert np.all(
                df.columns
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

        if name == "vc_curves":
            assert np.all(
                df.columns
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
        .add_pvss_halfsine_envelope(tstart=0, tstop=0.2),
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
            .add_pvss_halfsine_envelope(tstart=0, tstop=0.2)
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
        os.path.join(os.path.dirname(__file__), "test1.IDE"),
        os.path.join(os.path.dirname(__file__), "test2.IDE"),
        os.path.join(os.path.dirname(__file__), "test4.IDE"),
    ]

    calc_result = getdata_builder.aggregate_data(filenames)

    assert [x[0] for x in calc_result.dataframes[1:]] == [
        x[0] for x in getdata_builder._metrics_queue
    ]
    assert len(calc_result.dataframes[0][1]) == 3
    assert_output_is_valid(calc_result)


@pytest.fixture
def output_struct():
    data = []

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
    data.append(
        (
            "meta",
            pd.DataFrame.from_records(
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
            ),
        )
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
    data.append(
        (
            "psd",
            pd.DataFrame.from_records(
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
            ),
        )
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
    data.append(
        (
            "pvss",
            pd.DataFrame.from_records(
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
            ),
        )
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
    data.append(
        (
            "metrics",
            pd.DataFrame.from_records(
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
            ),
        )
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
    data.append(
        (
            "peaks",
            pd.DataFrame.from_records(
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
            ),
        )
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
    data.append(
        (
            "vc_curves",
            pd.DataFrame.from_records(
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
            ),
        )
    )

    result = endaq.batch.core.OutputStruct(data)
    assert_output_is_valid(result)

    return result


def test_output_to_csv_folder(output_struct):
    with tempfile.TemporaryDirectory() as dirpath:
        output_struct.to_csv_folder(dirpath)

        for name, df in output_struct.dataframes:
            filepath = os.path.join(dirpath, name + ".csv")
            assert os.path.isfile(filepath)

            read_result = pd.read_csv(
                filepath,
                **(dict(index_col="filename") if name == "meta" else {}),
            )
            assert df.astype(str).compare(read_result.astype(str)).size == 0


@pytest.mark.filterwarnings(
    "ignore:HTML plot for metrics not currently implemented:UserWarning"
)
def test_output_to_html_plots(output_struct):
    with tempfile.TemporaryDirectory() as dirpath:
        output_struct.to_html_plots(folder_path=dirpath, show=False, theme="endaq")

        for name, _df in output_struct.dataframes:
            # Not all dataframes get plotted
            if name in ("meta", "metrics"):
                continue

            filepath = os.path.join(dirpath, name + ".html")
            assert os.path.isfile(filepath)

            # can't do much else for validation...


@pytest.mark.parametrize(
    "filenames",
    [
        [
            os.path.join(os.path.dirname(__file__), "SSX70065.IDE"),
            os.path.join(os.path.dirname(__file__), "test1.IDE"),
            os.path.join(os.path.dirname(__file__), "test3.IDE"),
            os.path.join(os.path.dirname(__file__), "test5.IDE"),
            os.path.join(os.path.dirname(__file__), "GPS-Chick-Fil-A_003.IDE"),
            "https://info.endaq.com/hubfs/data/High-Drop.ide",
            "https://info.endaq.com/hubfs/data/Punching-Bag.ide",
        ],
        [
            os.path.join(os.path.dirname(__file__), "SSX70065.IDE"),
        ],
        [
            "https://info.endaq.com/hubfs/data/High-Drop.ide",
        ],
        [
            os.path.join(os.path.dirname(__file__), "SSX70065.IDE"),
            "https://info.endaq.com/hubfs/data/High-Drop.ide",
        ],
        [],
    ],
)
@pytest.mark.filterwarnings("ignore:empty frequency bins:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:no acceleration channel in:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:HTML plot for metrics not currently implemented:UserWarning"
)
def test_integration(filenames):
    output_struct = (
        endaq.batch.core.GetDataBuilder(accel_highpass_cutoff=1)
        .add_psd(freq_bin_width=1)
        .add_psd(freq_start_octave=2, bins_per_octave=12)
        .add_pvss(init_freq=1, bins_per_octave=12)
        .add_pvss_halfsine_envelope()
        .add_metrics(exclude=["RMS Velocity", "RMS Sound Pressure"])
        .add_peaks(margin_len=1000)
        .add_vc_curves(init_freq=1, bins_per_octave=3)
        .aggregate_data(filenames)
    )
    if len(filenames) == 0:
        assert output_struct is None
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_struct.to_csv_folder(tmp_dir)
        output_struct.to_html_plots(tmp_dir, show=False)
