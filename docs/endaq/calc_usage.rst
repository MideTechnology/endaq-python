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

Linearly & Octave Spaced
^^^^^^^^^^^^^^^
This presents some data from bearing tests explained in more detail in our `blog on calculating vibration metrics <https://blog.endaq.com/top-vibration-metrics-to-monitor-how-to-calculate-them>`_.

.. code:: python

   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   bearing = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

   #Calculate PSD with 1 Hz Bin Width
   psd = endaq.calc.psd.welch(bearing, bin_width=1)

   #Plot PSD
   fig1 = px.line(psd[10:5161]).update_layout(
       title_text='1 Hz PSD of Bearing Vibration',
       yaxis_title_text='Acceleration (g^2/Hz)',
       xaxis_title_text='Frequency (Hz)',
       xaxis_type='log',
       yaxis_type='log',
       legend_title_text='',
   )
   fig1.show()    

   #Calculate 1/3 Octave Spaced PSD    
   oct_psd = endaq.calc.psd.to_octave(psd, fstart=4, octave_bins=3)

   #Plot Octave PSD
   fig2 = px.line(oct_psd[10:5161]).update_layout(
       title_text='1/3 Octave PSD of Bearing Vibration',
       yaxis_title_text='Acceleration (g^2/Hz)',
       xaxis_title_text='Frequency (Hz)',
       xaxis_type='log',
       yaxis_type='log',
       legend_title_text='',
   )
   fig2.show()        
    
.. plotly::
   :fig-vars: fig1, fig2

   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   bearing = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

   #Calculate PSD with 1 Hz Bin Width
   psd = endaq.calc.psd.welch(bearing, bin_width=1)

   #Plot PSD
   fig1 = px.line(psd[10:5161]).update_layout(
       title_text='1 Hz PSD of Bearing Vibration',
       yaxis_title_text='Acceleration (g^2/Hz)',
       xaxis_title_text='Frequency (Hz)',
       xaxis_type='log',
       yaxis_type='log',
       legend_title_text='',
   )
   fig1.show()    

   #Calculate 1/3 Octave Spaced PSD    
   oct_psd = endaq.calc.psd.to_octave(psd, fstart=4, octave_bins=3)

   #Plot Octave PSD
   fig2 = px.line(oct_psd[10:5161]).update_layout(
       title_text='1/3 Octave PSD of Bearing Vibration',
       yaxis_title_text='Acceleration (g^2/Hz)',
       xaxis_title_text='Frequency (Hz)',
       xaxis_type='log',
       yaxis_type='log',
       legend_title_text='',
   )
   fig2.show()        
   
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
This presents some data from a motorcylce crash test that is explained in more detail in our `blog on shock response spectrums <https://blog.endaq.com/shock-analysis-response-spectrum-srs-pseudo-velocity-severity>`_.

.. code:: python

   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/Motorcycle-Car-Crash.ide')
   accel = endaq.ide.to_pandas(doc.channels[8], time_mode='seconds')[1137.4:1137.8]
   accel = accel - accel.median()

   #Calculate SRS
   freqs = endaq.calc.utils.logfreqs(accel, init_freq=1, bins_per_octave=12)
   srs = endaq.calc.shock.shock_spectrum(accel, freqs=freqs, damp=0.05, mode='srs')

   #Plot SRS
   fig1 = px.line(srs).update_layout(
       title_text='Shock Response Spectrum (SRS) of Motorcycle Crash',
       xaxis_title_text="Natural Frequency (Hz)",
       yaxis_title_text="Peak Acceleration (g)",
       legend_title_text='',
       xaxis_type="log",
       yaxis_type="log",
     )
   fig1.show()

   #Calculate PVSS
   pvss = endaq.calc.shock.shock_spectrum(accel, freqs=freqs, damp=0.05, mode='pvss')

   #Generate Half Sine Equivalents
   half_sine = endaq.calc.shock.enveloping_half_sine(pvss, damp=0.05)
   half_sine_pvss = endaq.calc.shock.shock_spectrum(half_sine.to_time_series(tstart=0,tstop=2), freqs=freqs, damp=0.05, mode='pvss')

   #Add to PVSS DataFrame
   half_sine_pvss.columns = half_sine.amplitude.astype(int).astype(str) + "g, " + np.round(half_sine.duration*1000,1).astype(str) + "ms"
   pvss = pd.concat([pvss,half_sine_pvss],axis=1)*9.81*39.37 #convert to in/s

   #Plot PVSS
   fig2 = px.line(pvss).update_layout(
       title_text='Psuedo Velocity Shock Spectrum (PVSS) of Motorcycle Crash w/ Half Sine Equivalents',
       xaxis_title_text="Natural Frequency (Hz)",
       yaxis_title_text="Psuedo Velocity (in/s)",
       legend_title_text='',
       xaxis_type="log",
       yaxis_type="log",
     )
   fig2.show()

.. plotly::
   :fig-vars: fig1, fig2
   
   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   doc = endaq.ide.get_doc('https://info.endaq.com/hubfs/data/Motorcycle-Car-Crash.ide')
   accel = endaq.ide.to_pandas(doc.channels[8], time_mode='seconds')[1137.4:1137.8]
   accel = accel - accel.median()

   #Calculate SRS
   freqs = endaq.calc.utils.logfreqs(accel, init_freq=1, bins_per_octave=12)
   srs = endaq.calc.shock.shock_spectrum(accel, freqs=freqs, damp=0.05, mode='srs')

   #Plot SRS
   fig1 = px.line(srs).update_layout(
       title_text='Shock Response Spectrum (SRS) of Motorcycle Crash',
       xaxis_title_text="Natural Frequency (Hz)",
       yaxis_title_text="Peak Acceleration (g)",
       legend_title_text='',
       xaxis_type="log",
       yaxis_type="log",
     )
   fig1.show()

   #Calculate PVSS
   pvss = endaq.calc.shock.shock_spectrum(accel, freqs=freqs, damp=0.05, mode='pvss')

   #Generate Half Sine Equivalents
   half_sine = endaq.calc.shock.enveloping_half_sine(pvss, damp=0.05)
   half_sine_pvss = endaq.calc.shock.shock_spectrum(half_sine.to_time_series(tstart=0,tstop=2), freqs=freqs, damp=0.05, mode='pvss')

   #Add to PVSS DataFrame
   half_sine_pvss.columns = half_sine.amplitude.astype(int).astype(str) + "g, " + np.round(half_sine.duration*1000,1).astype(str) + "ms"
   pvss = pd.concat([pvss,half_sine_pvss],axis=1)*9.81*39.37 #convert to in/s

   #Plot PVSS
   fig2 = px.line(pvss).update_layout(
       title_text='Psuedo Velocity Shock Spectrum (PVSS) of Motorcycle Crash w/ Half Sine Equivalents',
       xaxis_title_text="Natural Frequency (Hz)",
       yaxis_title_text="Psuedo Velocity (in/s)",
       legend_title_text='',
       xaxis_type="log",
       yaxis_type="log",
     )
   fig2.show()   
