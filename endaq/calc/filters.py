from __future__ import annotations

from typing import Optional, Union, Tuple, List
import functools

import pandas as pd
import numpy as np
import scipy.signal
import re

from endaq.calc import utils


def _get_filter_frequencies_type(low_cutoff, high_cutoff):
    """Get the filter type and cutoff frequency array."""
    cutoff_freqs: Union[float, Tuple[float, float]]
    filter_type: str

    if low_cutoff is not None and high_cutoff is not None:
        cutoff_freqs = (low_cutoff, high_cutoff)
        filter_type = "bandpass"
    elif low_cutoff is not None:
        cutoff_freqs = low_cutoff
        filter_type = "highpass"
    elif high_cutoff is not None:
        cutoff_freqs = high_cutoff
        filter_type = "lowpass"        
    else:
        return None, None
        
    return cutoff_freqs, filter_type        


def rolling_mean(
    df: pd.DataFrame,
    duration: float = 5.0
) -> pd.DataFrame:
    """
    Remove the rolling mean of an input time series dataframe

    :param df: the input data
    :param duration: the rolling window size in seconds to use
        - if `None` is given, the entire mean is removed 

    :return: a dataframe of the filtered data
    """
    if (duration is None):
        mean = df.mean()
    else:
        n = int(np.ceil(duration / utils.sample_spacing(df)) // 2 * 2 + 1)
        mean = df.rolling(n, min_periods=1, center=True).mean()    

    return df - mean   


def butterworth(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Butterworth filter to an array.

    This function uses Butterworth filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :return: the filtered data

    .. seealso::

        - `SciPy Butterworth filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
          Documentation for the butterworth filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs, filter_type = _get_filter_frequencies_type(low_cutoff, high_cutoff)

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.butter(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df


def bessel(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
    norm: typing.Literal["phase", "delay", "mag"] = "mag",
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Bessel filter to an array.

    This function uses Bessel filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :param norm: how to normalize relative to the critical frequency:
            * `"phase"` - "phase-matched" case which is the default in SciPy & MATLAB
            * `"delay"` - the "natural" type obtained by solving Bessel polynomials
            * `"mag"` - gain magnitude is -3 dB at the cutoff frequency, default for this implementation to match Butterworth

    :return: the filtered data

    .. seealso::

        - `SciPy Bessel filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html>`_
          Documentation for the Bessel filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs, filter_type = _get_filter_frequencies_type(low_cutoff, high_cutoff)

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.bessel(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
            norm=norm
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df  


def cheby1(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
    rp: float = 3.0,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Chebyshev type I filter to an array.

    This function uses Chebyshev type I filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :param rp: the maximum ripple allowed in the passband, specified in decibels

    :return: the filtered data

    .. seealso::

        - `SciPy Cheby1 filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html>`_
          Documentation for the Chebyshev type I filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs, filter_type = _get_filter_frequencies_type(low_cutoff, high_cutoff)

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.cheby1(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
            rp=rp
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df


def cheby2(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
    rs: float = 30.0,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Chebyshev type II filter to an array.

    This function uses Chebyshev type II filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :param rs: the minimum attenuation allowed in the stopband, specified in decibels

    :return: the filtered data

    .. seealso::

        - `SciPy Cheby2 filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html>`_
          Documentation for the Chebyshev type II filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs, filter_type = _get_filter_frequencies_type(low_cutoff, high_cutoff)

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.cheby2(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
            rs=rs
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df           


def ellip(
    df: pd.DataFrame,
    low_cutoff: Optional[float] = 1.0,
    high_cutoff: Optional[float] = None,
    half_order: int = 3,
    tukey_percent: float = 0.0,
    rp: float = 3.0,
    rs: float = 30.0,
) -> pd.DataFrame:
    """
    Apply a lowpass and/or a highpass Elliptic filter to an array.

    This function uses Elliptic filter designs, and implements the filter(s)
    as bi-directional digital biquad filters, split into second-order sections.

    :param df: the input data; cutoff frequencies are relative to the
        timestamps in `df.index`
    :param low_cutoff: the low-frequency cutoff, if any; frequencies below this
        value are rejected, and frequencies above this value are preserved
    :param high_cutoff: the high-frequency cutoff, if any; frequencies above
        this value are rejected, and frequencies below this value are preserved
    :param half_order: half of the order of the filter; higher orders provide
        more aggressive stopband reduction
    :param tukey_percent: the alpha parameter of a preconditioning Tukey filter;
        if 0 (default), no filter is applied
    :param rp: the maximum ripple allowed in the passband, specified in decibels
    :param rs: the minimum attenuation allowed in the stopband, specified in decibels

    :return: the filtered data

    .. seealso::

        - `SciPy Ellip filter design <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html>`_
          Documentation for the Elliptic filter design function.

        - `SciPy Tukey window <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html>`_
          Documentation for the Tukey window function used in preprocessing.
    """
    cutoff_freqs, filter_type = _get_filter_frequencies_type(low_cutoff, high_cutoff)

    if filter_type:
        dt = utils.sample_spacing(df)

        sos_coeffs = scipy.signal.ellip(
            N=half_order,
            Wn=cutoff_freqs,
            btype=filter_type,
            fs=1 / dt,
            output="sos",
            rs=rs,
            rp=rp
        )
        df = df.apply(functools.partial(scipy.signal.sosfiltfilt, sos_coeffs), axis=0)

    if tukey_percent > 0:
        tukey_window = scipy.signal.windows.tukey(len(df.index), alpha=tukey_percent)
        df = df.mul(tukey_window, axis="rows")

    return df            


def refine_acceleration(dfs: List[pd.DataFrame], part_number: str) -> pd.DataFrame:
    """
    Generates more accurate acceleration data by combining the two acceleration channels on
    an enDAQ device. This is done by filtering out inaccurate frequencies based off of known
    frequency curves, and averaging out data based on noise values.
    
    :param dfs: 2 panda dataframes with DateTimeIndex, converted by `endaq.ide.to_pandas`.
        Any non X / Y / Z acceleration channels will be dropped. The first dataframe should be 
        the main accelerometer (channel 8), and the second should be the secondary accelerometer,
        (channel 32 or channel 80).
    :param part_number: a string of the product name, that (for a doc `ds`) can be found by 
    `ds.recoderInfo['PartNumber']`.

    :return: a pandas Dataframe with the flattened acceleration data.
    """
    #removing non X/Y/Z acceleration channels
    for idx in range(len(dfs)):
        df = dfs[idx]
        df = df[df.columns[
            list(map(lambda x: x[0] in ['X', 'Y', 'Z'], 
                     df.columns))]].copy()
        for col in df.columns:
            df[col] = df[col] - np.mean(df[col])
        dfs[idx] = df
        

    aligned_dfs = utils.align_dataframes(dfs)
    sensors = re.findall(r"[E|D|R]\d+", part_number)
    if len(dfs) != 2 or len(dfs) != len(sensors):
        raise Exception("there needs to be exactly two dataframes and the part " \
        "number needs to contain exactly two sensor ids")

    #TODO: 0 mean everything ... easy way or the easy way.
    sensor_info = [_sensor_info(sensor) for sensor in sensors]
    noises = [si['noise'] for si in sensor_info]
    bounds = [(si['low_cutoff'], si['high_cutoff']) for si in sensor_info]
    dfs = [butterworth(df, low_cutoff= l_bound, high_cutoff= r_bound) 
                    for (df, (l_bound, r_bound)) in zip(aligned_dfs, bounds)]
    for df in dfs: df.columns = ['X', 'Y', 'Z']
    
    overlap = (max([bound[0] for bound in bounds]), min([bound[1] for bound in bounds]))
    
    hz_overlap_dfs = [butterworth(df, low_cutoff=overlap[0], high_cutoff=overlap[1])
                       for df in dfs] 
    
    averaged_df = _weighted_avg(hz_overlap_dfs, noises)
    hz_isolated_dfs = [] 
    for df, (l_bound, r_bound) in zip(dfs, bounds):
        isolated_df = df
        if l_bound == overlap[0]: butterworth(isolated_df, high_cutoff= l_bound)
        if r_bound == overlap[1]: butterworth(isolated_df, low_cutoff= r_bound)
        hz_isolated_dfs.append(isolated_df)

    return sum(hz_isolated_dfs + [averaged_df]) #+ is joining lists, not adding numbers

def _weighted_avg(dfs: List[pd.DataFrame], noise: List[float]):
    """
    Combines multiple dataframe into one by applying a weighted average base on noise in a linear-inverse fashion. 
    For sensors A and B, if the noise ratio is 2:1, the weighing ratio will be 1:2. Note that a value of 0 is 
    considered in the weighting.
    
    :param dfs: the dataframes to normalize. It is assumed that all dataframes have the same indecies. 
    :param noise: The noise value to the index-respective dataframe. Noise will be normalized.

    :return: a single dataframe with the weighted values
    """
    if len(dfs) != len(noise) or len(dfs) == 0:
        raise Exception("dataframes and noise need to have equal, non zero number of elements")
    #normalize the weightings. If there is a weighting with 0 noise, return that instead
    if 0 in noise:
        return dfs[noise.index(0)]
    noise = (np.array(noise) / sum(noise))[::-1]
    
    return pd.DataFrame(sum([n * d for n, d in zip(noise, dfs)]), 
                        columns = dfs[0].columns, index = dfs[0].index)
    

def _sensor_info(sensor: str) -> dict:
    """
    creates a dictionary with information relevant to :py:func:`refine_acceleration`, namely
    
    - sensor_type: Literal["D", "E", "R"]. Indicates if the sensor is digital, piezoelectric, 
        or piezoresistive
    - rating: the g rating of the sensor, or the maximum it can read while accurate
    - noise: float. Indicates the noise of the sensor when the response curve is flat
    - low_cutoff: the lowest Hz where the response curve is flat.
    - high_cutoff: the highest Hz where the response curve is flat.

    :return: a dictionary with the information stated above
    """
    sensor_type = sensor[0]
    sensor_rating = int(sensor[1:])

    match sensor_type:
        case "E":
            low = 10
            high = sensor_rating / 5
            noise = {25: 8E-4, 100: 3E-3, 500: 1.5E-2, 2000: 0.06}[sensor_rating]
        case "D":
            try:
                low, high = {16: (1,300), 40: (1, 100)}[sensor_rating]
                noise = {16: 4E-3, 40: 8E-5}[sensor_rating]
            except KeyError:
                raise Exception(f"rating {sensor_rating} not supported for digital IMUs")
        case "R":
            low = 1
            high = sensor_rating / 5
            noise = {100: 3E-3, 500: 1.5E-2, 2000: 0.06}
        case _:
            raise Exception(f"Sensor type {sensor_type} not recognized, should be one of E, D, R.")
        
    return {
        "sensor_type": sensor_type,
        "noise": noise, 
        "rating": sensor_rating, 
        "low_cutoff": low, 
        "high_cutoff": high,
        }

def _fftnoise(f):
    """
    Generate time series noise for a given range of frequencies with random phase using ifft.
    """
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def band_limited_noise(
    min_freq: float = 0.0,
    max_freq: float = None,
    duration: float = 1.0,
    sample_rate: float = 1000.0,
    norm: typing.Literal["rms", "peak"] = "peak",
) -> pd.DataFrame:
    """
    Generate a time series with noise in a defined frequency range.

    :param min_freq: minimum frequency (Hz) where noise starts, default to 0
    :param max_freq: maximum frequency (Hz) where noise ends, default to 1/2 the sample rate
    :param duration: the duration of the time series returned, in seconds
    :param sample_rate: sample rate (Hz) of the time series
    :param norm: how to normalize the amplitude so that one of the following is equal to 1:
            * `"rms"` - root mean square
            * `"peak"` - peak value, default

    :return: a dataframe of the generated time series

    .. seealso::

        - `stack overflow post <https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy>`_
          that inspired this function and shared code we based this function on.
    
    """
    samples = int(duration * sample_rate)
    freqs = np.abs(np.fft.fftfreq(samples, 1/sample_rate))
    f = np.zeros(samples)

    if (max_freq is None):
        max_freq = sample_rate/2
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1

    df = pd.DataFrame(_fftnoise(f), 
                        index=pd.Series(np.arange(samples)/sample_rate, name='Time (s)'),
                        columns=['Noise from '+str(np.round(min_freq,1))+' to '+str(np.round(max_freq,1))+' Hz'])
    
    if (norm == 'rms'):
        df = df / np.mean(df**2)**0.5
    else:
        df = df / df.abs().max()

    return df

