import os
from unittest import mock

import idelib
import numpy as np
import pandas as pd
import pytest
import hypothesis as hyp
import hypothesis.strategies as hyp_st
import hypothesis.extra.numpy as hyp_np

import endaq.batch.analyzer
from endaq.calc.stats import rms, L2_norm
from endaq.calc import to_dB


np.random.seed(0)


@pytest.fixture()
def ide_SSX70065():
    with idelib.importFile(os.path.join(os.path.dirname(__file__), "SSX70065.IDE")) as doc:
        yield doc


@pytest.fixture()
def analyzer_raw():
    analyzer_mock = mock.create_autospec(
        endaq.batch.analyzer.CalcCache, spec_set=False, instance=True
    )

    analyzer_mock.MPS2_TO_G = endaq.batch.analyzer.MPS2_TO_G
    analyzer_mock.MPS_TO_MMPS = endaq.batch.analyzer.MPS_TO_MMPS
    analyzer_mock.M_TO_MM = endaq.batch.analyzer.M_TO_MM
    analyzer_mock.PV_NATURAL_FREQS = endaq.batch.analyzer.CalcCache.PV_NATURAL_FREQS

    return analyzer_mock


@pytest.fixture()
def analyzer_bulk(analyzer_raw):
    analyzer_mock = analyzer_raw

    analyzer_mock._channels = {
        "acc": mock.Mock(axis_names=list("XYZ")),
        "gps": mock.Mock(axis_names=["Latitude", "Longitude"]),
        "spd": mock.Mock(axis_names=["Ground"]),
        "gyr": mock.Mock(axis_names=list("XYZ")),
        "mic": mock.Mock(axis_names=["Mic"]),
        "tmp": mock.Mock(axis_names=["Control"]),
        "pre": mock.Mock(axis_names=["Control"]),
    }

    analyzer_mock._accelerationFs = 3000
    analyzer_mock._accelerationData = pd.DataFrame(
        np.random.random((21, 3)),
        index=pd.Series(np.arange(21) / 3000, name="time"),
        columns=pd.Series(["X", "Y", "Z"], name="axis"),
    )
    analyzer_mock._accelerationResultantData = analyzer_mock._accelerationData.apply(
        L2_norm, axis="columns"
    ).to_frame()
    analyzer_mock._microphoneData = pd.DataFrame(
        np.random.random(21),
        index=pd.Series(np.arange(21) / 3000, name="time"),
        columns=pd.Series(["Mic"], name="axis"),
    )
    analyzer_mock._velocityData = pd.DataFrame(
        np.random.random((21, 3)),
        index=pd.Series(np.arange(21) / 3000, name="time"),
        columns=pd.Series(["X", "Y", "Z"], name="axis"),
    )
    analyzer_mock._displacementData = pd.DataFrame(
        np.random.random((21, 3)),
        index=pd.Series(np.arange(21) / 3000, name="time"),
        columns=pd.Series(["X", "Y", "Z"], name="axis"),
    )
    analyzer_mock._pressureData = pd.DataFrame(
        np.random.random(5),
        index=pd.Series(np.arange(5) / 5, name="time"),
        columns=pd.Series(["Control"], name="axis"),
    )
    analyzer_mock._temperatureData = pd.DataFrame(
        np.random.random(5),
        index=pd.Series(np.arange(5) / 5, name="time"),
        columns=pd.Series(["Control"], name="axis"),
    )
    analyzer_mock._gyroscopeData = pd.DataFrame(
        np.random.random(11),
        index=pd.Series(np.arange(11) / 5, name="time"),
        columns=pd.Series(["Gyro"], name="axis"),
    )

    return analyzer_mock


# ==============================================================================
# Analyzer class tests
# ==============================================================================


class TestAnalyzer:
    def test_from_ide_vs_from_literal(self, ide_SSX70065):
        dataset = ide_SSX70065
        calc_params = endaq.batch.analyzer.CalcParams(
            accel_start_time=None,
            accel_end_time=None,
            accel_start_margin=None,
            accel_end_margin=None,
            accel_highpass_cutoff=1,
            accel_integral_tukey_percent=0,
            accel_integral_zero="mean",
            psd_freq_bin_width=1,
            psd_window="hann",
            pvss_init_freq=1,
            pvss_bins_per_octave=12,
            vc_init_freq=1,
            vc_bins_per_octave=3,
        )

        dataset_cache = endaq.batch.analyzer.CalcCache.from_ide(dataset, calc_params)

        raw_cache = endaq.batch.analyzer.CalcCache.from_raw_data(
            [
                (
                    endaq.ide.to_pandas(dataset.channels[32], time_mode="timedelta"),
                    ("Acceleration", "g"),
                ),
                (
                    endaq.ide.to_pandas(
                        dataset.channels[36].subchannels[0], time_mode="timedelta"
                    ),
                    ("Pressure", "Pa"),
                ),
                (
                    endaq.ide.to_pandas(
                        dataset.channels[36].subchannels[1], time_mode="timedelta"
                    ),
                    ("Temperature", "°C"),
                ),
            ],
            calc_params,
        )

        assert set(dataset_cache._channels) == set(raw_cache._channels)

        for (ds_struct, raw_struct) in (
            (dataset_cache._channels[measure_key], raw_cache._channels[measure_key])
            for measure_key in dataset_cache._channels
        ):
            assert ds_struct.units == raw_struct.units
            pd.testing.assert_frame_equal(
                ds_struct.to_pandas(time_mode="timedelta"),
                raw_struct.to_pandas(time_mode="timedelta"),
            )

    @hyp.given(
        df=hyp_np.arrays(
            elements=hyp_st.floats(-1e7, 1e7),
            shape=(20, 2),
            dtype=np.float64,
        ).map(
            lambda array: pd.DataFrame(
                array, index=np.timedelta64(200, "ms") * np.arange(20)
            )
        ),
    )
    def test_accelerationData(self, df):
        calc_params = endaq.batch.analyzer.CalcParams(
            accel_start_time=None,
            accel_end_time=None,
            accel_start_margin=None,
            accel_end_margin=None,
            accel_highpass_cutoff=1,
            accel_integral_tukey_percent=0,
            accel_integral_zero="mean",
            psd_freq_bin_width=1,
            psd_window="hann",
            pvss_init_freq=1,
            pvss_bins_per_octave=12,
            vc_init_freq=1,
            vc_bins_per_octave=3,
        )
        data_cache = endaq.batch.analyzer.CalcCache.from_raw_data(
            [(df, ("Acceleration", "m/s\u00b2"))], calc_params
        )

        df_accel = endaq.calc.filters.butterworth(
            df, low_cutoff=calc_params.accel_highpass_cutoff
        )
        pd.testing.assert_frame_equal(data_cache._accelerationData, df_accel)

        (_df_accel, df_vel, df_displ) = endaq.calc.integrate.integrals(
            df_accel,
            n=2,
            zero=calc_params.accel_integral_zero,
            highpass_cutoff=calc_params.accel_highpass_cutoff,
            tukey_percent=calc_params.accel_integral_tukey_percent,
        )
        pd.testing.assert_frame_equal(data_cache._velocityData, df_vel)
        pd.testing.assert_frame_equal(data_cache._displacementData, df_displ)

    def test_accRMSFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.accRMSFull.func(analyzer_bulk)[
            "Resultant"
        ]
        expt_result = endaq.batch.analyzer.MPS2_TO_G * rms(
            analyzer_bulk._accelerationData.apply(L2_norm, axis="columns")
        )

        assert calc_result == pytest.approx(expt_result)

    def test_velRMSFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.velRMSFull.func(analyzer_bulk)[
            "Resultant"
        ]
        expt_result = endaq.batch.analyzer.MPS_TO_MMPS * rms(
            analyzer_bulk._velocityData.apply(L2_norm, axis="columns")
        )
        assert calc_result == pytest.approx(expt_result)

    def test_disRMSFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.disRMSFull.func(analyzer_bulk)[
            "Resultant"
        ]
        expt_result = endaq.batch.analyzer.M_TO_MM * rms(
            analyzer_bulk._displacementData.apply(L2_norm, axis="columns")
        )
        assert calc_result == pytest.approx(expt_result)

    def test_accPeakFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.accPeakFull.func(analyzer_bulk)[
            "Resultant"
        ]
        expt_result = endaq.batch.analyzer.MPS2_TO_G * rms(
            analyzer_bulk._accelerationData.apply(L2_norm, axis="columns").max()
        )

        assert calc_result == pytest.approx(expt_result)

    def test_pseudoVelPeakFull(self, analyzer_bulk):
        pass

    def test_gpsLocFull(self, analyzer_bulk):
        pass

    def test_gpsSpeedFull(self, analyzer_bulk):
        pass

    def test_gyroRMSFull(self, analyzer_bulk):
        pass

    def test_micDecibelsFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.micDecibelsFull.func(
            analyzer_bulk
        )["Mic"]
        expt_result = to_dB(rms(analyzer_bulk._microphoneData), reference="SPL")

        assert calc_result == pytest.approx(expt_result)

    def test_pressFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.pressFull.func(analyzer_bulk)[
            "Control"
        ]
        expt_result = analyzer_bulk._pressureData.mean()

        assert calc_result == pytest.approx(expt_result)

    def test_tempFull(self, analyzer_bulk):
        calc_result = endaq.batch.analyzer.CalcCache.tempFull.func(analyzer_bulk)[
            "Control"
        ]
        expt_result = analyzer_bulk._temperatureData.mean()

        assert calc_result == pytest.approx(expt_result)

    ###########################################################################
    # Live File Tests

    def testLiveFile(self, ide_SSX70065):
        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ide_SSX70065,
            endaq.batch.analyzer.CalcParams(
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_highpass_cutoff=1,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=1,
                psd_window="hann",
                pvss_init_freq=1,
                pvss_bins_per_octave=12,
                vc_init_freq=1,
                vc_bins_per_octave=3,
            ),
        )

        raw_accel = ide_SSX70065.channels[32].getSession().arrayValues()
        np.testing.assert_allclose(
            analyzer.accRMSFull["Resultant"],
            rms(L2_norm(raw_accel - raw_accel.mean(axis=-1, keepdims=True), axis=0)),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            analyzer.pressFull,
            1e-3
            * ide_SSX70065.channels[36]
            .subchannels[0]
            .getSession()
            .arrayValues()
            .mean(),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            analyzer.tempFull,
            ide_SSX70065.channels[36].subchannels[1].getSession().arrayValues().mean(),
            rtol=1e-4,
        )

    @pytest.mark.parametrize(
        "filename",
        [
            os.path.join(os.path.dirname(__file__), "test1.IDE"),
            os.path.join(os.path.dirname(__file__), "test2.IDE"),
        ],
    )
    def testLiveFiles1(self, filename):
        ds = idelib.importFile(filename)
        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ds,
            endaq.batch.analyzer.CalcParams(
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_highpass_cutoff=1,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=1,
                psd_window="hann",
                pvss_init_freq=1,
                pvss_bins_per_octave=12,
                vc_init_freq=1,
                vc_bins_per_octave=3,
            ),
        )

        raw_accel = ds.channels[8].getSession().arrayValues(subchannels=[0, 1, 2])
        np.testing.assert_allclose(
            analyzer.accRMSFull["Resultant"],
            rms(L2_norm(raw_accel - raw_accel.mean(axis=-1, keepdims=True), axis=0)),
            rtol=0.55,  # this is... probably a little too high...
        )
        audio_scale = 5.307530522779073
        np.testing.assert_allclose(
            analyzer.micDecibelsFull,
            to_dB(
                audio_scale
                * rms(ds.channels[8].subchannels[3].getSession().arrayValues()),
                reference="SPL",
            ),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            analyzer.gyroRMSFull["Resultant"],
            rms(L2_norm(ds.channels[84].getSession().arrayValues(), axis=0)),
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            analyzer.pressFull,
            1e-3 * ds.channels[36].subchannels[0].getSession().arrayValues().mean(),
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            analyzer.tempFull,
            ds.channels[36].subchannels[1].getSession().arrayValues().mean(),
            rtol=0.05,  # ...GEFGW?
        )
        np.testing.assert_allclose(
            analyzer.humidFull,
            ds.channels[59].subchannels[2].getSession().arrayValues().mean(),
        )

    @pytest.mark.parametrize(
        "filename",
        [
            os.path.join(os.path.dirname(__file__), "DAQ12006_000005.IDE"),
        ],
    )
    def testLiveFiles2(self, filename):
        """
        Checks that audio in units of Pascals are properly handled.
        """
        ds = idelib.importFile(filename)
        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ds,
            endaq.batch.analyzer.CalcParams(
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_highpass_cutoff=1,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=1,
                psd_window="hann",
                pvss_init_freq=1,
                pvss_bins_per_octave=12,
                vc_init_freq=1,
                vc_bins_per_octave=3,
            ),
        )

        np.testing.assert_allclose(
            analyzer.micDecibelsFull,
            to_dB(
                rms(ds.channels[8].subchannels[3].getSession().arrayValues()),
                reference="SPL",
            ),
            rtol=1e-3,
        )

    @pytest.mark.parametrize(
        "filename",
        [
            os.path.join(os.path.dirname(__file__), "test3.IDE"),
        ],
    )
    def testLiveFile3(self, filename):
        ds = idelib.importFile(filename)
        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ds,
            endaq.batch.analyzer.CalcParams(
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_highpass_cutoff=1,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=1,
                psd_window="hann",
                pvss_init_freq=1,
                pvss_bins_per_octave=12,
                vc_init_freq=1,
                vc_bins_per_octave=3,
            ),
        )

        ch_rot = ds.channels[84]
        np.testing.assert_allclose(
            analyzer.gyroRMSFull["Resultant"],
            rms(L2_norm(ch_rot.getSession().arrayValues(), axis=0)),
            rtol=0.005,
        )

    @pytest.mark.parametrize(
        "filename, sample_index",
        [
            (os.path.join(os.path.dirname(__file__), "test_GPS_2.IDE"), -2),
            (os.path.join(os.path.dirname(__file__), "test_GPS_3.IDE"), -4),
        ],
    )
    def testLiveFileGPS(self, filename, sample_index):
        ds = idelib.importFile(filename)
        analyzer = endaq.batch.analyzer.CalcCache.from_ide(
            ds,
            endaq.batch.analyzer.CalcParams(
                accel_start_time=None,
                accel_end_time=None,
                accel_start_margin=None,
                accel_end_margin=None,
                accel_highpass_cutoff=1,
                accel_integral_tukey_percent=0,
                accel_integral_zero="mean",
                psd_freq_bin_width=1,
                psd_window="hann",
                pvss_init_freq=1,
                pvss_bins_per_octave=12,
                vc_init_freq=1,
                vc_bins_per_octave=3,
            ),
        )

        ch_gps = ds.channels[88]
        # confirming channel format for gps file
        assert ch_gps.subchannels[0].name == "Latitude"
        assert ch_gps.subchannels[1].name == "Longitude"
        assert ch_gps.subchannels[3].name == "Ground Speed"

        assert tuple(analyzer.gpsLocFull) == tuple(
            ch_gps.getSession().arrayValues()[[0, 1], sample_index],
        )
        # Resampling throws off mean calculation
        # -> use resampled data for comparsion
        # gps_speed = analyzer.MPS_TO_KMPH * ch_gps.getSession().arrayValues(subchannels=[3])
        gps_speed = analyzer._gpsSpeedData
        np.testing.assert_allclose(
            analyzer.gpsSpeedFull,
            np.mean(gps_speed[gps_speed != 0]),
        )
