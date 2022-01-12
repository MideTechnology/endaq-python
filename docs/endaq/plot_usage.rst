``endaq.plot`` Usage Examples
=============================

For these examples we assume there is a Pandas DataFrame named ``df``
which has it’s index as time stamps and it’s one column being sensor
values (e.g. x-axis acceleration, or pressure). It also assumes there is
a Pandas DataFrame ``attribute_df`` which contains all the attribute
data about various data files. More information can be found about how
to get this data from enDAQ IDE files in the `endaq-cloud
readme <https://github.com/MideTechnology/endaq-python/tree/main/endaq/cloud>`__.

.. code:: python

   from endaq.plot import octave_spectrogram, multi_file_plot_attributes, octave_psd_bar_plot
   from endaq.plot.utilities import set_theme

Setting The Aesthetic Theme
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   set_theme(theme='endaq')

Creating Spectrograms With Octave Spaced Frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   data_df, fig = octave_spectrogram(df, window=.15)
   fig.show()

.. figure:: https://i.imgur.com/929aszu.png
   :alt: Spectrogram With Octave Spaced Frequencies

   Spectrogram With Octave Spaced Frequencies

Creating PSD Bar Plots With Octave Spaced Frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   fig = octave_psd_bar_plot(df, yaxis_title="Magnitude")
   fig.show()

.. figure:: https://i.imgur.com/ueqcVTQ.png
   :alt: PSD Bar Plot With Octave Spaced Frequencies

   PSD Bar Plot With Octave Spaced Frequencies

Plot Attributes In Figure With Subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   fig = multi_file_plot_attributes(attribute_df)
   fig.show()

.. figure:: https://i.imgur.com/5Yy4DN7.png
   :alt: Attributes Plotted As Subplots

   Attributes Plotted As Subplots

Other Links
-----------

-  the endaq package - https://github.com/MideTechnology/endaq-python
-  the enDAQ homepage - https://endaq.com/
