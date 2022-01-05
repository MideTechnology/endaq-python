=============================
``endaq.calc`` Usage Examples
=============================


Filters
~~~~~~~
.. code:: python

   import endaq
   import matplotlib.pyplot as plt

   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc('https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8],time_mode='seconds')

   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)

   ax = df_accel_highpass['Z (100g)'].plot(xlabel='time (s)')

   df_accel_lowpass['Z (100g)'].plot(ax=ax)

   plt.legend(['highpass, 1Hz cutoff', 'lowpass, 100Hz cutoff'])
   plt.show()

.. plot::

   import endaq
   import matplotlib.pyplot as plt

   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc('https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8],time_mode='seconds')

   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)

   ax = df_accel_highpass['Z (100g)'].plot(xlabel='time (s)')

   df_accel_lowpass['Z (100g)'].plot(ax=ax)

   plt.legend(['highpass, 1Hz cutoff', 'lowpass, 100Hz cutoff'])
   plt.show()

Integration
~~~~~~~~~~~

.. code:: python

   dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)

PSD
~~~

Linearly-spaced
^^^^^^^^^^^^^^^

.. code:: python

   df_accel_psd = endaq.calc.psd.welch(df_accel, bin_width=1/11)

Octave-spaced
^^^^^^^^^^^^^

.. code:: python

   df_accel_psd_oct = endaq.calc.psd.to_octave(df_accel_psd, fstart=1, octave_bins=3)

Derivatives & Integrals
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   df_vel_psd = endaq.calc.psd.differentiate(df_accel_psd, n=-1)
   df_jerk_psd = endaq.calc.psd.differentiate(df_accel_psd, n=1)

Vibration Criterion (VC) Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   df_accel_vc = endaq.calc.psd.vc_curves(df_accel_psd, fstart=1, octave_bins=3)

Shock Analysis
~~~~~~~~~~~~~~

.. code:: python

   df_accel_pvss = endaq.calc.shock.shock_spectrum(df_accel, freqs=2 ** np.arange(-10, 13, 0.25), damp=0.05, mode="pvss")
   df_accel_srs = endaq.calc.shock.shock_spectrum(df_accel, freqs=[1, 10, 100, 1000], damp=0.05, mode="srs")

Shock Characterization: Half-Sine-Wave Pulse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   half_sine_params = endaq.calc.shock.enveloping_half_sine(df_accel_pvss, damp=0.05)
