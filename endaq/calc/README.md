# endaq-calc - a computational backend for vibration analysis

endaq-calc is a package comprising a collection of common calculations for vibration analysis. It leverages the standard Python scientific stack (NumPy, SciPy, Pandas) in order to enable engineers to perform domain-specific calculations in a few lines of code, without having to first master Python and its scientific stack in their entireties.

endaq-calc is a sub-package of the larger enDAQ ecosystem. See [the endaq package](https://github.com/MideTechnology/endaq-python) for more details.

## Installation

endaq-calc is available on PYPI via `pip`:

    pip install endaq-calc

For the most recent features that are still under development, you can also use `pip` to install endaq-calc directly from GitHub:

    pip install git+https://github.com/MideTechnology/endaq-python-calc.git@development

## Usage Examples

``` python
import endaq.calc.filters
import endaq.calc.integrate
import endaq.calc.psd
import endaq.calc.shock
```

### Filters
``` python
df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)
```

### Integration
``` python
dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)
```

### PSD

#### Linearly-spaced
``` python
df_accel_psd = endaq.calc.psd.welch(df_accel, bin_width=1/11)
```

#### Octave-spaced
``` python
df_accel_psd_oct = endaq.calc.psd.to_octave(df_accel_psd, fstart=1, octave_bins=3)
```

#### Derivatives & Integrals {#derivatives--integrals}
``` python
df_vel_psd = endaq.calc.psd.differentiate(df_accel_psd, n=-1)
df_jerk_psd = endaq.calc.psd.differentiate(df_accel_psd, n=1)
```

#### Vibration Criterion (VC) Curves
``` python
df_accel_vc = endaq.calc.psd.vc_curves(df_accel_psd, fstart=1, octave_bins=3)
```

### Shock Analysis
``` python
df_accel_pvss = endaq.calc.shock.shock_spectrum(df_accel, freqs=2 ** np.arange(-10, 13, 0.25), damp=0.05, mode="pvss")
df_accel_srs = endaq.calc.shock.shock_spectrum(df_accel, freqs=[1, 10, 100, 1000], damp=0.05, mode="srs")
```

#### Shock Characterization: Half-Sine-Wave Pulse
``` python
half_sine_params = endaq.calc.shock.enveloping_half_sine(df_accel_pvss, damp=0.05)
```

## Other Links
- the endaq package - https://github.com/MideTechnology/endaq-python
- the enDAQ homepage - https://endaq.com/
