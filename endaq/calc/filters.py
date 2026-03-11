from __future__ import annotations

from typing import Optional, Union, Tuple, List
import functools

import pandas as pd
import numpy as np
import scipy.signal

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


def refine_acceleration(
        dfs : List[pd.DataFrame], 
        product_name: Optional[str],
        *,
        deg: Optional[int] = None,
        window_length: Optional[int] = 10,
        step = 3
        ) -> pd.DataFrame:
    """
    Combines mutliple acceleration channels to create a with a flat frequency response curve
    using a modified savitzky-golay filter. Note that this will remove 2 * py:param:`window_length`
    datapoints, py:param:`window_length` from each side.

    :param dfs: a List of dataframes to combine, all of which has a 'X', 'Y', and 'Z' channel.
    :param product_name: the name of the enDAQ device used to record the dataframes, which can 
        be found through doc.recorderInfo['ProductName']. If None, the error will come from the 
        worst case of error between the sensor types of the rating.

    :param deg: a keyword-only argument that sets the degree of the polynomial
        in the salgov for polynomial fitting. If left as None, a degree will be calculated
    :param window_length: a keyword-only argument that sets the size of the window used in
        the salgov for polynomial fitting.
    :param step: a keyword-only argument that sets how many points are generated at each
        step of the savitzky-golay filter. This parameter cannot be larger than
        py:param:`window_length`

    :return: a DataFrame, with columns ['X (adjusted)', 'Y (adjusted)', 'Z (adjusted)'], with
        each column containing their respective data.
    """

    #drop any channels that don't match the correct regex

    #create / apply noise column

    #combine dataframes using pd.concat

    #convert to numeric, drop any NaN / inf values

    #apply weighted_savgol_filter()

    #combine to pd.DataFrame

    pass

def weighted_savgol_filter(x: List[float],
                           y: List[float], 
                           deg: int, 
                           window_length: int = 1, 
                           w: Optional[List[float]] = None, 
                           step : int = 1) -> pd.Series:
    """
    A variation on the savgol filter, who adjusts points by fitting them to a windowed best fit 
    curve of a given degree. This implementation adds :py:param:`weight` and :py:param:`step`
    parameters, which allow the filter to have bias towards lower noise variables, and perform less
    calculation for higher step values.

    :param x: with n datapoints.
    :param y: with n datapoints.
    :param deg: the degree of the polynomial to fit to.
    :param w: a List of weights respective to the y point in the same index with n datapoints.
        If None is given, all points will be weighted equal, like a standard savgol implementation.
    :param window_length: the size of the window, dictating how many datapoints will be fit to the 
        polynomial.
    :param step: dictates how many datapoints are generated from the best fit polynomial, 
        between 1 and n.

    :return: a pandas series where the index is x, and the values are the adjusted y-points.
    """ 
    #validity checks
    if len(x) != len(y) or len(x) != len(w):
        raise ValueError("x, y, and weights all need to share equal number of points")
    
    if len(x) <= window_length:
        raise ValueError("the number of datapoints has to be greater than the window size")
    
    new_x = []
    new_y = []
    
    start_idx = window_length
    end_idx = start_idx + window_length

    while end_idx < len(x):
        x_temp = x[start_idx:end_idx]
        y_temp = y[start_idx:end_idx]
        w_temp = w[start_idx:end_idx]
        poly_fit = np.polyfit(
            x = x_temp, 
            y = y_temp, 
            deg = deg if deg is not None else np.ceil(window_length / 2),
            w = w_temp,
        )
        mid_point = x[int(np.ceil((start_idx + end_idx)/ 2))]
        new_x.append(mid_point)
        new_y.append(
            np.sum([(mid_point ** (i - 1)) * poly_fit[-1 * i]
                    for i in range(len(poly_fit), 0, -1)]))
        
        start_idx += step
        end_idx += step
    return pd.Series(new_y, index = new_x)

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

