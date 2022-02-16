############################
``endaq.ide`` Usage Examples
############################

Note: For brevity, the following examples assume everything has been imported
from :py:mod:`endaq.ide`:

.. code:: python3

    from endaq.ide import *

Opening IDE files: :py:func:`endaq.ide.get_doc()`
-------------------------------------------------

:py:mod:`endaq.ide` includes a convenient shortcut for importing IDE data:
:py:func:`~endaq.ide.get_doc()`. It can load data from local files, or read data directly
from a URL.

.. code:: python3

    doc = get_doc("tests/test.ide")
    doc1 = get_doc("https://info.endaq.com/hubfs/data/surgical-instrument.ide")

IDE files can be retrieved directly from Google Drive using a Drive
'sharable link' URL. The file must be set to allow access to "Anyone
with the link."

.. code:: python3

    doc2 = get_doc("https://drive.google.com/file/d/1t3JqbZGhuZbIK9agH24YZIdVE26-NOF5/view?usp=sharing")

Whether opening a local file or a URL, :py:func:`~endaq.ide.get_doc()` can be used to
import only a specific interval by way of its ``start`` and ``end``
parameters:

.. code:: python3

    doc3 = get_doc("tests/test.ide", start="5s", end="10s")

Summarizing IDE files: :py:func:`endaq.ide.get_channel_table()`
---------------------------------------------------------------

Once an IDE file has been loaded, :py:func:`~endaq.ide.get_channel_table()` will
retrieve basic summary information about its contents.

.. code:: python3

    get_channel_table(doc)




.. raw:: html

    <style type="text/css">
    </style>
    <table id="T_d2cbb_">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th class="col_heading level0 col0" >channel</th>
          <th class="col_heading level0 col1" >name</th>
          <th class="col_heading level0 col2" >type</th>
          <th class="col_heading level0 col3" >units</th>
          <th class="col_heading level0 col4" >start</th>
          <th class="col_heading level0 col5" >end</th>
          <th class="col_heading level0 col6" >duration</th>
          <th class="col_heading level0 col7" >samples</th>
          <th class="col_heading level0 col8" >rate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_d2cbb_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_d2cbb_row0_col0" class="data row0 col0" >32.0</td>
          <td id="T_d2cbb_row0_col1" class="data row0 col1" >X (16g)</td>
          <td id="T_d2cbb_row0_col2" class="data row0 col2" >Acceleration</td>
          <td id="T_d2cbb_row0_col3" class="data row0 col3" >g</td>
          <td id="T_d2cbb_row0_col4" class="data row0 col4" >00:00.0952</td>
          <td id="T_d2cbb_row0_col5" class="data row0 col5" >00:19.0012</td>
          <td id="T_d2cbb_row0_col6" class="data row0 col6" >00:18.0059</td>
          <td id="T_d2cbb_row0_col7" class="data row0 col7" >7113</td>
          <td id="T_d2cbb_row0_col8" class="data row0 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_d2cbb_row1_col0" class="data row1 col0" >32.1</td>
          <td id="T_d2cbb_row1_col1" class="data row1 col1" >Y (16g)</td>
          <td id="T_d2cbb_row1_col2" class="data row1 col2" >Acceleration</td>
          <td id="T_d2cbb_row1_col3" class="data row1 col3" >g</td>
          <td id="T_d2cbb_row1_col4" class="data row1 col4" >00:00.0952</td>
          <td id="T_d2cbb_row1_col5" class="data row1 col5" >00:19.0012</td>
          <td id="T_d2cbb_row1_col6" class="data row1 col6" >00:18.0059</td>
          <td id="T_d2cbb_row1_col7" class="data row1 col7" >7113</td>
          <td id="T_d2cbb_row1_col8" class="data row1 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_d2cbb_row2_col0" class="data row2 col0" >32.2</td>
          <td id="T_d2cbb_row2_col1" class="data row2 col1" >Z (16g)</td>
          <td id="T_d2cbb_row2_col2" class="data row2 col2" >Acceleration</td>
          <td id="T_d2cbb_row2_col3" class="data row2 col3" >g</td>
          <td id="T_d2cbb_row2_col4" class="data row2 col4" >00:00.0952</td>
          <td id="T_d2cbb_row2_col5" class="data row2 col5" >00:19.0012</td>
          <td id="T_d2cbb_row2_col6" class="data row2 col6" >00:18.0059</td>
          <td id="T_d2cbb_row2_col7" class="data row2 col7" >7113</td>
          <td id="T_d2cbb_row2_col8" class="data row2 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_d2cbb_row3_col0" class="data row3 col0" >80.0</td>
          <td id="T_d2cbb_row3_col1" class="data row3 col1" >X (8g)</td>
          <td id="T_d2cbb_row3_col2" class="data row3 col2" >Acceleration</td>
          <td id="T_d2cbb_row3_col3" class="data row3 col3" >g</td>
          <td id="T_d2cbb_row3_col4" class="data row3 col4" >00:00.0948</td>
          <td id="T_d2cbb_row3_col5" class="data row3 col5" >00:19.0013</td>
          <td id="T_d2cbb_row3_col6" class="data row3 col6" >00:18.0064</td>
          <td id="T_d2cbb_row3_col7" class="data row3 col7" >9070</td>
          <td id="T_d2cbb_row3_col8" class="data row3 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_d2cbb_row4_col0" class="data row4 col0" >80.1</td>
          <td id="T_d2cbb_row4_col1" class="data row4 col1" >Y (8g)</td>
          <td id="T_d2cbb_row4_col2" class="data row4 col2" >Acceleration</td>
          <td id="T_d2cbb_row4_col3" class="data row4 col3" >g</td>
          <td id="T_d2cbb_row4_col4" class="data row4 col4" >00:00.0948</td>
          <td id="T_d2cbb_row4_col5" class="data row4 col5" >00:19.0013</td>
          <td id="T_d2cbb_row4_col6" class="data row4 col6" >00:18.0064</td>
          <td id="T_d2cbb_row4_col7" class="data row4 col7" >9070</td>
          <td id="T_d2cbb_row4_col8" class="data row4 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_d2cbb_row5_col0" class="data row5 col0" >80.2</td>
          <td id="T_d2cbb_row5_col1" class="data row5 col1" >Z (8g)</td>
          <td id="T_d2cbb_row5_col2" class="data row5 col2" >Acceleration</td>
          <td id="T_d2cbb_row5_col3" class="data row5 col3" >g</td>
          <td id="T_d2cbb_row5_col4" class="data row5 col4" >00:00.0948</td>
          <td id="T_d2cbb_row5_col5" class="data row5 col5" >00:19.0013</td>
          <td id="T_d2cbb_row5_col6" class="data row5 col6" >00:18.0064</td>
          <td id="T_d2cbb_row5_col7" class="data row5 col7" >9070</td>
          <td id="T_d2cbb_row5_col8" class="data row5 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_d2cbb_row6_col0" class="data row6 col0" >36.0</td>
          <td id="T_d2cbb_row6_col1" class="data row6 col1" >Pressure/Temperature:00</td>
          <td id="T_d2cbb_row6_col2" class="data row6 col2" >Pressure</td>
          <td id="T_d2cbb_row6_col3" class="data row6 col3" >Pa</td>
          <td id="T_d2cbb_row6_col4" class="data row6 col4" >00:00.0945</td>
          <td id="T_d2cbb_row6_col5" class="data row6 col5" >00:19.0175</td>
          <td id="T_d2cbb_row6_col6" class="data row6 col6" >00:18.0230</td>
          <td id="T_d2cbb_row6_col7" class="data row6 col7" >20</td>
          <td id="T_d2cbb_row6_col8" class="data row6 col8" >1.10 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_d2cbb_row7_col0" class="data row7 col0" >36.1</td>
          <td id="T_d2cbb_row7_col1" class="data row7 col1" >Pressure/Temperature:01</td>
          <td id="T_d2cbb_row7_col2" class="data row7 col2" >Temperature</td>
          <td id="T_d2cbb_row7_col3" class="data row7 col3" >°C</td>
          <td id="T_d2cbb_row7_col4" class="data row7 col4" >00:00.0945</td>
          <td id="T_d2cbb_row7_col5" class="data row7 col5" >00:19.0175</td>
          <td id="T_d2cbb_row7_col6" class="data row7 col6" >00:18.0230</td>
          <td id="T_d2cbb_row7_col7" class="data row7 col7" >20</td>
          <td id="T_d2cbb_row7_col8" class="data row7 col8" >1.10 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row8" class="row_heading level0 row8" >8</th>
          <td id="T_d2cbb_row8_col0" class="data row8 col0" >70.0</td>
          <td id="T_d2cbb_row8_col1" class="data row8 col1" >X</td>
          <td id="T_d2cbb_row8_col2" class="data row8 col2" >Quaternion</td>
          <td id="T_d2cbb_row8_col3" class="data row8 col3" >q</td>
          <td id="T_d2cbb_row8_col4" class="data row8 col4" >00:01.0132</td>
          <td id="T_d2cbb_row8_col5" class="data row8 col5" >00:18.0954</td>
          <td id="T_d2cbb_row8_col6" class="data row8 col6" >00:17.0821</td>
          <td id="T_d2cbb_row8_col7" class="data row8 col7" >1755</td>
          <td id="T_d2cbb_row8_col8" class="data row8 col8" >98.47 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row9" class="row_heading level0 row9" >9</th>
          <td id="T_d2cbb_row9_col0" class="data row9 col0" >70.1</td>
          <td id="T_d2cbb_row9_col1" class="data row9 col1" >Y</td>
          <td id="T_d2cbb_row9_col2" class="data row9 col2" >Quaternion</td>
          <td id="T_d2cbb_row9_col3" class="data row9 col3" >q</td>
          <td id="T_d2cbb_row9_col4" class="data row9 col4" >00:01.0132</td>
          <td id="T_d2cbb_row9_col5" class="data row9 col5" >00:18.0954</td>
          <td id="T_d2cbb_row9_col6" class="data row9 col6" >00:17.0821</td>
          <td id="T_d2cbb_row9_col7" class="data row9 col7" >1755</td>
          <td id="T_d2cbb_row9_col8" class="data row9 col8" >98.47 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row10" class="row_heading level0 row10" >10</th>
          <td id="T_d2cbb_row10_col0" class="data row10 col0" >70.2</td>
          <td id="T_d2cbb_row10_col1" class="data row10 col1" >Z</td>
          <td id="T_d2cbb_row10_col2" class="data row10 col2" >Quaternion</td>
          <td id="T_d2cbb_row10_col3" class="data row10 col3" >q</td>
          <td id="T_d2cbb_row10_col4" class="data row10 col4" >00:01.0132</td>
          <td id="T_d2cbb_row10_col5" class="data row10 col5" >00:18.0954</td>
          <td id="T_d2cbb_row10_col6" class="data row10 col6" >00:17.0821</td>
          <td id="T_d2cbb_row10_col7" class="data row10 col7" >1755</td>
          <td id="T_d2cbb_row10_col8" class="data row10 col8" >98.47 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row11" class="row_heading level0 row11" >11</th>
          <td id="T_d2cbb_row11_col0" class="data row11 col0" >70.3</td>
          <td id="T_d2cbb_row11_col1" class="data row11 col1" >W</td>
          <td id="T_d2cbb_row11_col2" class="data row11 col2" >Quaternion</td>
          <td id="T_d2cbb_row11_col3" class="data row11 col3" >q</td>
          <td id="T_d2cbb_row11_col4" class="data row11 col4" >00:01.0132</td>
          <td id="T_d2cbb_row11_col5" class="data row11 col5" >00:18.0954</td>
          <td id="T_d2cbb_row11_col6" class="data row11 col6" >00:17.0821</td>
          <td id="T_d2cbb_row11_col7" class="data row11 col7" >1755</td>
          <td id="T_d2cbb_row11_col8" class="data row11 col8" >98.47 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row12" class="row_heading level0 row12" >12</th>
          <td id="T_d2cbb_row12_col0" class="data row12 col0" >59.0</td>
          <td id="T_d2cbb_row12_col1" class="data row12 col1" >Control Pad Pressure</td>
          <td id="T_d2cbb_row12_col2" class="data row12 col2" >Pressure</td>
          <td id="T_d2cbb_row12_col3" class="data row12 col3" >Pa</td>
          <td id="T_d2cbb_row12_col4" class="data row12 col4" >00:00.0979</td>
          <td id="T_d2cbb_row12_col5" class="data row12 col5" >00:18.0910</td>
          <td id="T_d2cbb_row12_col6" class="data row12 col6" >00:17.0931</td>
          <td id="T_d2cbb_row12_col7" class="data row12 col7" >180</td>
          <td id="T_d2cbb_row12_col8" class="data row12 col8" >10.04 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row13" class="row_heading level0 row13" >13</th>
          <td id="T_d2cbb_row13_col0" class="data row13 col0" >59.1</td>
          <td id="T_d2cbb_row13_col1" class="data row13 col1" >Control Pad Temperature</td>
          <td id="T_d2cbb_row13_col2" class="data row13 col2" >Temperature</td>
          <td id="T_d2cbb_row13_col3" class="data row13 col3" >°C</td>
          <td id="T_d2cbb_row13_col4" class="data row13 col4" >00:00.0979</td>
          <td id="T_d2cbb_row13_col5" class="data row13 col5" >00:18.0910</td>
          <td id="T_d2cbb_row13_col6" class="data row13 col6" >00:17.0931</td>
          <td id="T_d2cbb_row13_col7" class="data row13 col7" >180</td>
          <td id="T_d2cbb_row13_col8" class="data row13 col8" >10.04 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row14" class="row_heading level0 row14" >14</th>
          <td id="T_d2cbb_row14_col0" class="data row14 col0" >76.0</td>
          <td id="T_d2cbb_row14_col1" class="data row14 col1" >Lux</td>
          <td id="T_d2cbb_row14_col2" class="data row14 col2" >Light</td>
          <td id="T_d2cbb_row14_col3" class="data row14 col3" >Ill</td>
          <td id="T_d2cbb_row14_col4" class="data row14 col4" >00:00.0000</td>
          <td id="T_d2cbb_row14_col5" class="data row14 col5" >00:18.0737</td>
          <td id="T_d2cbb_row14_col6" class="data row14 col6" >00:18.0737</td>
          <td id="T_d2cbb_row14_col7" class="data row14 col7" >71</td>
          <td id="T_d2cbb_row14_col8" class="data row14 col8" >3.79 Hz</td>
        </tr>
        <tr>
          <th id="T_d2cbb_level0_row15" class="row_heading level0 row15" >15</th>
          <td id="T_d2cbb_row15_col0" class="data row15 col0" >76.1</td>
          <td id="T_d2cbb_row15_col1" class="data row15 col1" >UV</td>
          <td id="T_d2cbb_row15_col2" class="data row15 col2" >Light</td>
          <td id="T_d2cbb_row15_col3" class="data row15 col3" >Index</td>
          <td id="T_d2cbb_row15_col4" class="data row15 col4" >00:00.0000</td>
          <td id="T_d2cbb_row15_col5" class="data row15 col5" >00:18.0737</td>
          <td id="T_d2cbb_row15_col6" class="data row15 col6" >00:18.0737</td>
          <td id="T_d2cbb_row15_col7" class="data row15 col7" >71</td>
          <td id="T_d2cbb_row15_col8" class="data row15 col8" >3.79 Hz</td>
        </tr>
      </tbody>
    </table>




The results can be filtered by :doc:`measurement type <ide_measurement>`:

.. code:: python3

    get_channel_table(doc, ACCELERATION)




.. raw:: html

    <style type="text/css">
    </style>
    <table id="T_9f9cf_">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th class="col_heading level0 col0" >channel</th>
          <th class="col_heading level0 col1" >name</th>
          <th class="col_heading level0 col2" >type</th>
          <th class="col_heading level0 col3" >units</th>
          <th class="col_heading level0 col4" >start</th>
          <th class="col_heading level0 col5" >end</th>
          <th class="col_heading level0 col6" >duration</th>
          <th class="col_heading level0 col7" >samples</th>
          <th class="col_heading level0 col8" >rate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_9f9cf_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_9f9cf_row0_col0" class="data row0 col0" >32.0</td>
          <td id="T_9f9cf_row0_col1" class="data row0 col1" >X (16g)</td>
          <td id="T_9f9cf_row0_col2" class="data row0 col2" >Acceleration</td>
          <td id="T_9f9cf_row0_col3" class="data row0 col3" >g</td>
          <td id="T_9f9cf_row0_col4" class="data row0 col4" >00:00.0952</td>
          <td id="T_9f9cf_row0_col5" class="data row0 col5" >00:19.0012</td>
          <td id="T_9f9cf_row0_col6" class="data row0 col6" >00:18.0059</td>
          <td id="T_9f9cf_row0_col7" class="data row0 col7" >7113</td>
          <td id="T_9f9cf_row0_col8" class="data row0 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_9f9cf_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_9f9cf_row1_col0" class="data row1 col0" >32.1</td>
          <td id="T_9f9cf_row1_col1" class="data row1 col1" >Y (16g)</td>
          <td id="T_9f9cf_row1_col2" class="data row1 col2" >Acceleration</td>
          <td id="T_9f9cf_row1_col3" class="data row1 col3" >g</td>
          <td id="T_9f9cf_row1_col4" class="data row1 col4" >00:00.0952</td>
          <td id="T_9f9cf_row1_col5" class="data row1 col5" >00:19.0012</td>
          <td id="T_9f9cf_row1_col6" class="data row1 col6" >00:18.0059</td>
          <td id="T_9f9cf_row1_col7" class="data row1 col7" >7113</td>
          <td id="T_9f9cf_row1_col8" class="data row1 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_9f9cf_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_9f9cf_row2_col0" class="data row2 col0" >32.2</td>
          <td id="T_9f9cf_row2_col1" class="data row2 col1" >Z (16g)</td>
          <td id="T_9f9cf_row2_col2" class="data row2 col2" >Acceleration</td>
          <td id="T_9f9cf_row2_col3" class="data row2 col3" >g</td>
          <td id="T_9f9cf_row2_col4" class="data row2 col4" >00:00.0952</td>
          <td id="T_9f9cf_row2_col5" class="data row2 col5" >00:19.0012</td>
          <td id="T_9f9cf_row2_col6" class="data row2 col6" >00:18.0059</td>
          <td id="T_9f9cf_row2_col7" class="data row2 col7" >7113</td>
          <td id="T_9f9cf_row2_col8" class="data row2 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_9f9cf_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_9f9cf_row3_col0" class="data row3 col0" >80.0</td>
          <td id="T_9f9cf_row3_col1" class="data row3 col1" >X (8g)</td>
          <td id="T_9f9cf_row3_col2" class="data row3 col2" >Acceleration</td>
          <td id="T_9f9cf_row3_col3" class="data row3 col3" >g</td>
          <td id="T_9f9cf_row3_col4" class="data row3 col4" >00:00.0948</td>
          <td id="T_9f9cf_row3_col5" class="data row3 col5" >00:19.0013</td>
          <td id="T_9f9cf_row3_col6" class="data row3 col6" >00:18.0064</td>
          <td id="T_9f9cf_row3_col7" class="data row3 col7" >9070</td>
          <td id="T_9f9cf_row3_col8" class="data row3 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_9f9cf_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_9f9cf_row4_col0" class="data row4 col0" >80.1</td>
          <td id="T_9f9cf_row4_col1" class="data row4 col1" >Y (8g)</td>
          <td id="T_9f9cf_row4_col2" class="data row4 col2" >Acceleration</td>
          <td id="T_9f9cf_row4_col3" class="data row4 col3" >g</td>
          <td id="T_9f9cf_row4_col4" class="data row4 col4" >00:00.0948</td>
          <td id="T_9f9cf_row4_col5" class="data row4 col5" >00:19.0013</td>
          <td id="T_9f9cf_row4_col6" class="data row4 col6" >00:18.0064</td>
          <td id="T_9f9cf_row4_col7" class="data row4 col7" >9070</td>
          <td id="T_9f9cf_row4_col8" class="data row4 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_9f9cf_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_9f9cf_row5_col0" class="data row5 col0" >80.2</td>
          <td id="T_9f9cf_row5_col1" class="data row5 col1" >Z (8g)</td>
          <td id="T_9f9cf_row5_col2" class="data row5 col2" >Acceleration</td>
          <td id="T_9f9cf_row5_col3" class="data row5 col3" >g</td>
          <td id="T_9f9cf_row5_col4" class="data row5 col4" >00:00.0948</td>
          <td id="T_9f9cf_row5_col5" class="data row5 col5" >00:19.0013</td>
          <td id="T_9f9cf_row5_col6" class="data row5 col6" >00:18.0064</td>
          <td id="T_9f9cf_row5_col7" class="data row5 col7" >9070</td>
          <td id="T_9f9cf_row5_col8" class="data row5 col8" >502.09 Hz</td>
        </tr>
      </tbody>
    </table>




Measurement types can be combined to retrieve more than one:

.. code:: python3

    get_channel_table(doc, ACCELERATION+TEMPERATURE)




.. raw:: html

    <style type="text/css">
    </style>
    <table id="T_68598_">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th class="col_heading level0 col0" >channel</th>
          <th class="col_heading level0 col1" >name</th>
          <th class="col_heading level0 col2" >type</th>
          <th class="col_heading level0 col3" >units</th>
          <th class="col_heading level0 col4" >start</th>
          <th class="col_heading level0 col5" >end</th>
          <th class="col_heading level0 col6" >duration</th>
          <th class="col_heading level0 col7" >samples</th>
          <th class="col_heading level0 col8" >rate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_68598_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_68598_row0_col0" class="data row0 col0" >32.0</td>
          <td id="T_68598_row0_col1" class="data row0 col1" >X (16g)</td>
          <td id="T_68598_row0_col2" class="data row0 col2" >Acceleration</td>
          <td id="T_68598_row0_col3" class="data row0 col3" >g</td>
          <td id="T_68598_row0_col4" class="data row0 col4" >00:00.0952</td>
          <td id="T_68598_row0_col5" class="data row0 col5" >00:19.0012</td>
          <td id="T_68598_row0_col6" class="data row0 col6" >00:18.0059</td>
          <td id="T_68598_row0_col7" class="data row0 col7" >7113</td>
          <td id="T_68598_row0_col8" class="data row0 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_68598_row1_col0" class="data row1 col0" >32.1</td>
          <td id="T_68598_row1_col1" class="data row1 col1" >Y (16g)</td>
          <td id="T_68598_row1_col2" class="data row1 col2" >Acceleration</td>
          <td id="T_68598_row1_col3" class="data row1 col3" >g</td>
          <td id="T_68598_row1_col4" class="data row1 col4" >00:00.0952</td>
          <td id="T_68598_row1_col5" class="data row1 col5" >00:19.0012</td>
          <td id="T_68598_row1_col6" class="data row1 col6" >00:18.0059</td>
          <td id="T_68598_row1_col7" class="data row1 col7" >7113</td>
          <td id="T_68598_row1_col8" class="data row1 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_68598_row2_col0" class="data row2 col0" >32.2</td>
          <td id="T_68598_row2_col1" class="data row2 col1" >Z (16g)</td>
          <td id="T_68598_row2_col2" class="data row2 col2" >Acceleration</td>
          <td id="T_68598_row2_col3" class="data row2 col3" >g</td>
          <td id="T_68598_row2_col4" class="data row2 col4" >00:00.0952</td>
          <td id="T_68598_row2_col5" class="data row2 col5" >00:19.0012</td>
          <td id="T_68598_row2_col6" class="data row2 col6" >00:18.0059</td>
          <td id="T_68598_row2_col7" class="data row2 col7" >7113</td>
          <td id="T_68598_row2_col8" class="data row2 col8" >393.86 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_68598_row3_col0" class="data row3 col0" >80.0</td>
          <td id="T_68598_row3_col1" class="data row3 col1" >X (8g)</td>
          <td id="T_68598_row3_col2" class="data row3 col2" >Acceleration</td>
          <td id="T_68598_row3_col3" class="data row3 col3" >g</td>
          <td id="T_68598_row3_col4" class="data row3 col4" >00:00.0948</td>
          <td id="T_68598_row3_col5" class="data row3 col5" >00:19.0013</td>
          <td id="T_68598_row3_col6" class="data row3 col6" >00:18.0064</td>
          <td id="T_68598_row3_col7" class="data row3 col7" >9070</td>
          <td id="T_68598_row3_col8" class="data row3 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_68598_row4_col0" class="data row4 col0" >80.1</td>
          <td id="T_68598_row4_col1" class="data row4 col1" >Y (8g)</td>
          <td id="T_68598_row4_col2" class="data row4 col2" >Acceleration</td>
          <td id="T_68598_row4_col3" class="data row4 col3" >g</td>
          <td id="T_68598_row4_col4" class="data row4 col4" >00:00.0948</td>
          <td id="T_68598_row4_col5" class="data row4 col5" >00:19.0013</td>
          <td id="T_68598_row4_col6" class="data row4 col6" >00:18.0064</td>
          <td id="T_68598_row4_col7" class="data row4 col7" >9070</td>
          <td id="T_68598_row4_col8" class="data row4 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_68598_row5_col0" class="data row5 col0" >80.2</td>
          <td id="T_68598_row5_col1" class="data row5 col1" >Z (8g)</td>
          <td id="T_68598_row5_col2" class="data row5 col2" >Acceleration</td>
          <td id="T_68598_row5_col3" class="data row5 col3" >g</td>
          <td id="T_68598_row5_col4" class="data row5 col4" >00:00.0948</td>
          <td id="T_68598_row5_col5" class="data row5 col5" >00:19.0013</td>
          <td id="T_68598_row5_col6" class="data row5 col6" >00:18.0064</td>
          <td id="T_68598_row5_col7" class="data row5 col7" >9070</td>
          <td id="T_68598_row5_col8" class="data row5 col8" >502.09 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_68598_row6_col0" class="data row6 col0" >36.1</td>
          <td id="T_68598_row6_col1" class="data row6 col1" >Pressure/Temperature:01</td>
          <td id="T_68598_row6_col2" class="data row6 col2" >Temperature</td>
          <td id="T_68598_row6_col3" class="data row6 col3" >°C</td>
          <td id="T_68598_row6_col4" class="data row6 col4" >00:00.0945</td>
          <td id="T_68598_row6_col5" class="data row6 col5" >00:19.0175</td>
          <td id="T_68598_row6_col6" class="data row6 col6" >00:18.0230</td>
          <td id="T_68598_row6_col7" class="data row6 col7" >20</td>
          <td id="T_68598_row6_col8" class="data row6 col8" >1.10 Hz</td>
        </tr>
        <tr>
          <th id="T_68598_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_68598_row7_col0" class="data row7 col0" >59.1</td>
          <td id="T_68598_row7_col1" class="data row7 col1" >Control Pad Temperature</td>
          <td id="T_68598_row7_col2" class="data row7 col2" >Temperature</td>
          <td id="T_68598_row7_col3" class="data row7 col3" >°C</td>
          <td id="T_68598_row7_col4" class="data row7 col4" >00:00.0979</td>
          <td id="T_68598_row7_col5" class="data row7 col5" >00:18.0910</td>
          <td id="T_68598_row7_col6" class="data row7 col6" >00:17.0931</td>
          <td id="T_68598_row7_col7" class="data row7 col7" >180</td>
          <td id="T_68598_row7_col8" class="data row7 col8" >10.04 Hz</td>
        </tr>
      </tbody>
    </table>




Information about a specific interval can be retrieved by using the
``start`` and/or ``end`` arguments. Note that due to different sampling
rates, the start and end times for slower channels may not precisely
match the specified ``start`` or ``end``.

.. code:: python3

    get_channel_table(doc, ACCELERATION+TEMPERATURE, start="0:05", end="0:10")




.. raw:: html

    <style type="text/css">
    </style>
    <table id="T_6ade9_">
      <thead>
        <tr>
          <th class="blank level0" >&nbsp;</th>
          <th class="col_heading level0 col0" >channel</th>
          <th class="col_heading level0 col1" >name</th>
          <th class="col_heading level0 col2" >type</th>
          <th class="col_heading level0 col3" >units</th>
          <th class="col_heading level0 col4" >start</th>
          <th class="col_heading level0 col5" >end</th>
          <th class="col_heading level0 col6" >duration</th>
          <th class="col_heading level0 col7" >samples</th>
          <th class="col_heading level0 col8" >rate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th id="T_6ade9_level0_row0" class="row_heading level0 row0" >0</th>
          <td id="T_6ade9_row0_col0" class="data row0 col0" >32.0</td>
          <td id="T_6ade9_row0_col1" class="data row0 col1" >X (16g)</td>
          <td id="T_6ade9_row0_col2" class="data row0 col2" >Acceleration</td>
          <td id="T_6ade9_row0_col3" class="data row0 col3" >g</td>
          <td id="T_6ade9_row0_col4" class="data row0 col4" >00:05.0000</td>
          <td id="T_6ade9_row0_col5" class="data row0 col5" >00:10.0001</td>
          <td id="T_6ade9_row0_col6" class="data row0 col6" >00:05.0000</td>
          <td id="T_6ade9_row0_col7" class="data row0 col7" >1969</td>
          <td id="T_6ade9_row0_col8" class="data row0 col8" >393.75 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row1" class="row_heading level0 row1" >1</th>
          <td id="T_6ade9_row1_col0" class="data row1 col0" >32.1</td>
          <td id="T_6ade9_row1_col1" class="data row1 col1" >Y (16g)</td>
          <td id="T_6ade9_row1_col2" class="data row1 col2" >Acceleration</td>
          <td id="T_6ade9_row1_col3" class="data row1 col3" >g</td>
          <td id="T_6ade9_row1_col4" class="data row1 col4" >00:05.0000</td>
          <td id="T_6ade9_row1_col5" class="data row1 col5" >00:10.0001</td>
          <td id="T_6ade9_row1_col6" class="data row1 col6" >00:05.0000</td>
          <td id="T_6ade9_row1_col7" class="data row1 col7" >1969</td>
          <td id="T_6ade9_row1_col8" class="data row1 col8" >393.75 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row2" class="row_heading level0 row2" >2</th>
          <td id="T_6ade9_row2_col0" class="data row2 col0" >32.2</td>
          <td id="T_6ade9_row2_col1" class="data row2 col1" >Z (16g)</td>
          <td id="T_6ade9_row2_col2" class="data row2 col2" >Acceleration</td>
          <td id="T_6ade9_row2_col3" class="data row2 col3" >g</td>
          <td id="T_6ade9_row2_col4" class="data row2 col4" >00:05.0000</td>
          <td id="T_6ade9_row2_col5" class="data row2 col5" >00:10.0001</td>
          <td id="T_6ade9_row2_col6" class="data row2 col6" >00:05.0000</td>
          <td id="T_6ade9_row2_col7" class="data row2 col7" >1969</td>
          <td id="T_6ade9_row2_col8" class="data row2 col8" >393.75 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row3" class="row_heading level0 row3" >3</th>
          <td id="T_6ade9_row3_col0" class="data row3 col0" >80.0</td>
          <td id="T_6ade9_row3_col1" class="data row3 col1" >X (8g)</td>
          <td id="T_6ade9_row3_col2" class="data row3 col2" >Acceleration</td>
          <td id="T_6ade9_row3_col3" class="data row3 col3" >g</td>
          <td id="T_6ade9_row3_col4" class="data row3 col4" >00:05.0000</td>
          <td id="T_6ade9_row3_col5" class="data row3 col5" >00:10.0001</td>
          <td id="T_6ade9_row3_col6" class="data row3 col6" >00:05.0000</td>
          <td id="T_6ade9_row3_col7" class="data row3 col7" >2510</td>
          <td id="T_6ade9_row3_col8" class="data row3 col8" >501.98 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row4" class="row_heading level0 row4" >4</th>
          <td id="T_6ade9_row4_col0" class="data row4 col0" >80.1</td>
          <td id="T_6ade9_row4_col1" class="data row4 col1" >Y (8g)</td>
          <td id="T_6ade9_row4_col2" class="data row4 col2" >Acceleration</td>
          <td id="T_6ade9_row4_col3" class="data row4 col3" >g</td>
          <td id="T_6ade9_row4_col4" class="data row4 col4" >00:05.0000</td>
          <td id="T_6ade9_row4_col5" class="data row4 col5" >00:10.0001</td>
          <td id="T_6ade9_row4_col6" class="data row4 col6" >00:05.0000</td>
          <td id="T_6ade9_row4_col7" class="data row4 col7" >2510</td>
          <td id="T_6ade9_row4_col8" class="data row4 col8" >501.98 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row5" class="row_heading level0 row5" >5</th>
          <td id="T_6ade9_row5_col0" class="data row5 col0" >80.2</td>
          <td id="T_6ade9_row5_col1" class="data row5 col1" >Z (8g)</td>
          <td id="T_6ade9_row5_col2" class="data row5 col2" >Acceleration</td>
          <td id="T_6ade9_row5_col3" class="data row5 col3" >g</td>
          <td id="T_6ade9_row5_col4" class="data row5 col4" >00:05.0000</td>
          <td id="T_6ade9_row5_col5" class="data row5 col5" >00:10.0001</td>
          <td id="T_6ade9_row5_col6" class="data row5 col6" >00:05.0000</td>
          <td id="T_6ade9_row5_col7" class="data row5 col7" >2510</td>
          <td id="T_6ade9_row5_col8" class="data row5 col8" >501.98 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row6" class="row_heading level0 row6" >6</th>
          <td id="T_6ade9_row6_col0" class="data row6 col0" >36.1</td>
          <td id="T_6ade9_row6_col1" class="data row6 col1" >Pressure/Temperature:01</td>
          <td id="T_6ade9_row6_col2" class="data row6 col2" >Temperature</td>
          <td id="T_6ade9_row6_col3" class="data row6 col3" >°C</td>
          <td id="T_6ade9_row6_col4" class="data row6 col4" >00:04.0954</td>
          <td id="T_6ade9_row6_col5" class="data row6 col5" >00:10.0966</td>
          <td id="T_6ade9_row6_col6" class="data row6 col6" >00:06.0011</td>
          <td id="T_6ade9_row6_col7" class="data row6 col7" >6</td>
          <td id="T_6ade9_row6_col8" class="data row6 col8" >1.00 Hz</td>
        </tr>
        <tr>
          <th id="T_6ade9_level0_row7" class="row_heading level0 row7" >7</th>
          <td id="T_6ade9_row7_col0" class="data row7 col0" >59.1</td>
          <td id="T_6ade9_row7_col1" class="data row7 col1" >Control Pad Temperature</td>
          <td id="T_6ade9_row7_col2" class="data row7 col2" >Temperature</td>
          <td id="T_6ade9_row7_col3" class="data row7 col3" >°C</td>
          <td id="T_6ade9_row7_col4" class="data row7 col4" >00:05.0086</td>
          <td id="T_6ade9_row7_col5" class="data row7 col5" >00:10.0095</td>
          <td id="T_6ade9_row7_col6" class="data row7 col6" >00:05.0008</td>
          <td id="T_6ade9_row7_col7" class="data row7 col7" >50</td>
          <td id="T_6ade9_row7_col8" class="data row7 col8" >9.98 Hz</td>
        </tr>
      </tbody>
    </table>




Extracting intervals: :py:func:`endaq.ide.extract_time()`
---------------------------------------------------------

A portion of an IDE file can be saved to another, new IDE. The source
can be a local filename or an opened IDE (from a file or URL).

.. code:: python3

    extract_time("tests/test.ide", "doc_extracted.ide", start="0:05", end="0:10")
    extract_time(doc1, "doc1_extracted.ide", start="0:05", end="0:10")

Additional sample IDE recording files
-------------------------------------

Here are a number of example IDE files, which may be used with
:py:mod:`endaq.ide`:

.. code:: python3

    file_urls = ['https://info.endaq.com/hubfs/data/surgical-instrument.ide',
                 'https://info.endaq.com/hubfs/data/97c3990f-Drive-Home_70-1616632444.ide',
                 'https://info.endaq.com/hubfs/data/High-Drop.ide',
                 'https://info.endaq.com/hubfs/data/HiTest-Shock.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home_01.ide',
                 'https://info.endaq.com/hubfs/data/Tower-of-Terror.ide',
                 'https://info.endaq.com/hubfs/data/Punching-Bag.ide',
                 'https://info.endaq.com/hubfs/data/Gun-Stock.ide',
                 'https://info.endaq.com/hubfs/data/Seat-Base_21.ide',
                 'https://info.endaq.com/hubfs/data/Seat-Top_09.ide',
                 'https://info.endaq.com/hubfs/data/Bolted.ide',
                 'https://info.endaq.com/hubfs/data/Motorcycle-Car-Crash.ide',
                 'https://info.endaq.com/hubfs/data/train-passing.ide',
                 'https://info.endaq.com/hubfs/data/baseball.ide',
                 'https://info.endaq.com/hubfs/data/Clean-Room-VC.ide',
                 'https://info.endaq.com/hubfs/data/enDAQ_Cropped.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home_07.ide',
                 'https://info.endaq.com/hubfs/data/ford_f150.ide',
                 'https://info.endaq.com/hubfs/data/Drive-Home.ide',
                 'https://info.endaq.com/hubfs/data/Mining-Data.ide',
                 'https://info.endaq.com/hubfs/data/Mide-Airport-Drive-Lexus-Hybrid-Dash-W8.ide']

These can be directly read from ``endaq.com`` using :py:func:`endaq.ide.get_doc()`,
as previously described.