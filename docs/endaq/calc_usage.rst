=============================
``endaq.calc`` Usage Examples
=============================


Filters
~~~~~~~
.. code:: python

   import plotly.express as px

   import endaq
   endaq.plot.utilities.set_theme('endaq_light')


   # get accelerometer data from https://info.endaq.com/hubfs/100Hz_shake_cal.ide
   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )

   # create a new dataframe filtered with a high-pass butterworth with a cutoff frequency of 1 Hz
   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_highpass.columns = ['1Hz high-pass filter']

   # create a new dataframe filtered with a low-pass butterworth with a cutoff frequency of 100 Hz
   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)
   df_accel_lowpass.columns = ['100Hz low-pass filter']

   # merge the data into a single dataframe for plotting
   df_accel = df_accel.join(df_accel_highpass, how='left')
   df_accel = df_accel.join(df_accel_lowpass, how='left')

   # plot everything on the same axes
   fig1 = px.line(
           df_accel,
           x=df_accel.index,
           y=df_accel.columns,
           labels=
               {
                   "timestamp": "time [s]",
                   "value": "Acceleration [g]",
               },
       )
   fig1.show()

.. plotly::
   :fig-vars: fig1

   import plotly.express as px

   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   # get accelerometer data from https://info.endaq.com/hubfs/100Hz_shake_cal.ide
   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )

   # create a new dataframe filtered with a high-pass butterworth with a cutoff frequency of 1 Hz
   df_accel_highpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=1, high_cutoff=None)
   df_accel_highpass.columns = ['1Hz high-pass filter']

   # create a new dataframe filtered with a low-pass butterworth with a cutoff frequency of 100 Hz
   df_accel_lowpass = endaq.calc.filters.butterworth(df_accel, low_cutoff=None, high_cutoff=100)
   df_accel_lowpass.columns = ['100Hz low-pass filter']

   # merge the data into a single dataframe for plotting
   df_accel = df_accel.join(df_accel_highpass, how='left')
   df_accel = df_accel.join(df_accel_lowpass, how='left')

   # plot everything on the same axes
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
   endaq.plot.utilities.set_theme('endaq_light')


   # get accelerometer data from https://info.endaq.com/hubfs/100Hz_shake_cal.ide and convert from g to m/s^2
   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )*9.81  # g to m/s^2

   # generate the velocity and position by integrating after filtering out low-frequency content below 1 Hz
   dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)[1]
   dfs_integrate_2 = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=1.0, tukey_percent=0.05)[2]
   df_accel.columns = ['acceleration']
   dfs_integrate.columns = ['velocity']
   dfs_integrate_2.columns = ['displacement']

   # combine dataframes for plotting and adjust the scale so everything fits well
   df_accel = df_accel.join(dfs_integrate*1e3, how='left')  # m/s -> mm/s
   df_accel = df_accel.join(dfs_integrate_2*1e6, how='left')  # m -> μm

   # plot everything on the same axes
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
   endaq.plot.utilities.set_theme('endaq_light')


   # get accelerometer data from https://info.endaq.com/hubfs/100Hz_shake_cal.ide and convert from g to m/s^2
   df_accel = endaq.ide.to_pandas(endaq.ide.get_doc(
           'https://info.endaq.com/hubfs/100Hz_shake_cal.ide').channels[8].subchannels[2],
           time_mode='seconds',
       )*9.81  # g to m/s^2

   # generate the velocity and position by integrating after filtering out low-frequency content below 1 Hz
   dfs_integrate = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=10.0, tukey_percent=0.05)[1]
   dfs_integrate_2 = endaq.calc.integrate.integrals(df_accel, n=2, highpass_cutoff=10.0, tukey_percent=0.05)[2]
   df_accel.columns = ['acceleration']
   dfs_integrate.columns = ['velocity']
   dfs_integrate_2.columns = ['position']

   # combine dataframes for plotting and adjust the scale so everything fits well
   df_accel = df_accel.join(dfs_integrate*1e3, how='left')  # m/s -> mm/s
   df_accel = df_accel.join(dfs_integrate_2*1e6, how='left')  # m -> μm

   # plot everything on the same axes
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
^^^^^^^^^^^^^^^^^^^^^^^^
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

RMS from PSD
^^^^^^^^^^^^^^^^^^^^^^^^
Calculating RMS of arbitrary frequency ranges is made possible with specifying ``scaling='parseval'`` in the :py:func:`~endaq.calc.psd.welch()` method and then using the :py:func:`~endaq.calc.psd.to_jagged()` method. Note that the overall RMS is the collective RMS of the individual ranges.

.. code:: python   

   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

   #Compute RMS of Time History
   rms_time = np.mean(accel**2)**0.5

   #Define Frequency Breakpoints
   freqs = [0.1, 65, 300, 1400, 5999]
   labels = [str(freqs[i]) + ' to ' + str(freqs[i+1]) for i in range(len(freqs)-1)]

   #Compute RMS for Frequency Ranges Using PSD Functions
   parseval = endaq.calc.psd.welch(accel, scaling='parseval', bin_width=0.5)
   rms_psd_breaks = endaq.calc.psd.to_jagged(parseval, freqs, agg='sum')**0.5

   #Plot Bar Chart of Frequency Ranges
   rms_psd_breaks.index = pd.Series(labels, name='Range (Hz)')
   fig1 = px.bar(rms_psd_breaks, barmode='group').update_layout(
       yaxis_title_text='Acceleration RMS (g)',
       legend_title_text='',
       title_text='RMS in Frequency Ranges'
   )
   fig1.show()

   #Compare the RMS Calculation from Time Domain to One Using PSD
     #Note that the Overall RMS is the Collective RMS of the Individual Ranges
   fig2 = px.bar(
       {
        'Time Domain':rms_time,
        'PSD w/ Breaks':np.sum(rms_psd_breaks**2)**0.5
       },
       barmode='group').update_layout(
       xaxis_title_text='',
       yaxis_title_text='Acceleration RMS (g)',
       legend_title_text='',
       title_text='RMS from Time vs from PSD'
   )
   fig2.show()     
   
.. plotly::
   :fig-vars: fig1, fig2
   
   import plotly.express as px
   import pandas as pd
   import endaq
   endaq.plot.utilities.set_theme('endaq_light')

   #Get Acceleration Data
   accel = pd.read_csv('https://info.endaq.com/hubfs/Plots/bearing_data.csv', index_col=0)

   #Compute RMS of Time History
   rms_time = np.mean(accel**2)**0.5

   #Define Frequency Breakpoints
   freqs = [0.1, 65, 300, 1400, 5999]
   labels = [str(freqs[i]) + ' to ' + str(freqs[i+1]) for i in range(len(freqs)-1)]

   #Compute RMS for Frequency Ranges Using PSD Functions
   parseval = endaq.calc.psd.welch(accel, scaling='parseval', bin_width=0.5)
   rms_psd_breaks = endaq.calc.psd.to_jagged(parseval, freqs, agg='sum')**0.5

   #Plot Bar Chart of Frequency Ranges
   rms_psd_breaks.index = pd.Series(labels, name='Range (Hz)')
   fig1 = px.bar(rms_psd_breaks, barmode='group').update_layout(
       yaxis_title_text='Acceleration RMS (g)',
       legend_title_text='',
       title_text='RMS in Frequency Ranges'
   )

   #Compare the RMS Calculation from Time Domain to One Using PSD
     #Note that the Overall RMS is the Collective RMS of the Individual Ranges
   fig2 = px.bar(
       {
        'Time Domain':rms_time,
        'PSD w/ Breaks':np.sum(rms_psd_breaks**2)**0.5
       },
       barmode='group').update_layout(
       xaxis_title_text='',
       yaxis_title_text='Acceleration RMS (g)',
       legend_title_text='',
       title_text='RMS from Time vs from PSD'
   )
   
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
This presents some data from a motorcycle crash test that is explained in more detail in our `blog on shock response spectrums <https://blog.endaq.com/shock-analysis-response-spectrum-srs-pseudo-velocity-severity>`_.

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
       title_text='PVSS w/ Half Sine Equivalents',
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
       title_text='PVSS w/ Half Sine Equivalents',
       xaxis_title_text="Natural Frequency (Hz)",
       yaxis_title_text="Psuedo Velocity (in/s)",
       legend_title_text='',
       xaxis_type="log",
       yaxis_type="log",
     )
