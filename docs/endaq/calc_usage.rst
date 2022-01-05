=============================
``endaq.calc`` Usage Examples
=============================


Filters
~~~~~~~
.. code:: python

   import plotly.express as px

   import endaq


   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )

   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_highpass.columns = ['1Hz high-pass filter']

   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)
   df_accel_lowpass.columns = ['100Hz low-pass filter']

   df_accel = df_accel.join(df_accel_highpass, how='left')
   df_accel = df_accel.join(df_accel_lowpass, how='left')

   fig1 = px.line(
           df_accel,
           x=df_accel.index,
           y=df_accel.columns,
           labels=
               {
                   "timestamp": "time [s]",
                   "value": "Acceleration [m/s^2]",
               },
       )
   fig1.show()

.. plotly::
   :fig-vars: fig1

   import plotly.express as px

   import endaq


   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )

   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_highpass.columns = ['1Hz high-pass filter']

   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)
   df_accel_lowpass.columns = ['100Hz low-pass filter']

   df_accel = df_accel.join(df_accel_highpass, how='left')
   df_accel = df_accel.join(df_accel_lowpass, how='left')

   fig1 = px.line(
           df_accel,
           x=df_accel.index,
           y=df_accel.columns,
           labels=
               {
                   "timestamp": "time [s]",
                   "value": "Acceleration [m/s^2]",
               },
       )


Integration
~~~~~~~~~~~

.. code:: python

   import plotly.express as px

   import endaq


   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )*9.81  # g to m/s^2

   dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)[1]
   dfs_integrate_2 = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)[2]
   df_accel.columns = ['acceleration']
   dfs_integrate.columns = ['velocity']
   dfs_integrate_2.columns = ['position']

   df_accel = df_accel.join(dfs_integrate*1e3, how='left')
   df_accel = df_accel.join(dfs_integrate_2*1e6, how='left')

   fig1 = px.line(
           df_accel,
           x=df_accel.index,
           y=df_accel.columns[::-1],
           labels=
               {
                   "timestamp": "time [s]",
                   "value": "Acceleration [m/s^2], Velocity [mm/s], Displacement [μm]",
               },
       )
   fig1.show()


.. plotly::
   :fig-vars: fig1

   import plotly.express as px

   import endaq


   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )*9.81  # g to m/s^2

   dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=10.0, tukey_percent=0.05)[1]
   dfs_integrate_2 = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=10.0, tukey_percent=0.05)[2]
   df_accel.columns = ['acceleration']
   dfs_integrate.columns = ['velocity']
   dfs_integrate_2.columns = ['position']

   df_accel = df_accel.join(dfs_integrate*1e3, how='left')
   df_accel = df_accel.join(dfs_integrate_2*1e6, how='left')

   fig1 = px.line(
           df_accel,
           x=df_accel.index,
           y=df_accel.columns[::-1],
           labels=
               {
                   "timestamp": "time [s]",
                   "value": "Acceleration [m/s^2], Velocity [mm/s], Displacement [μm]",
               },
       )


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
