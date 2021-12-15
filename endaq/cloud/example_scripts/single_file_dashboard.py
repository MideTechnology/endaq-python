def single_file_analysis_script(files: list,
                                file_download_url: str,
                                channel: str = '40g DC Acceleration',
                                subchannel: str = 'Z (40g)') -> str:
    """
    Creates a cloud dashboard (for use in cloud.endaq.com custom report generation) for plotting and exploring the
    data in the most recently uploaded file to the enDAQ Cloud.  It creates a PSD plot with octave spaced frequency
    bins, a plot of the channel data around the peak value, a spectrogram of subchannel data that's frequency bins are
    octave spaced, and a rolling envelope of every sensors data.

    :param files: A `list` of JSON blobs of recording file data as it's given in a custom report in cloud.endaq.com
    :param file_download_url: The download URL for the IDE file most recently uploaded to cloud.endaq.com
    :param channel: The name of the channel to focus the analysis on
    :param subchannel: The name of the subchannel to focus some of the analysis on
    :return: The json based string which is to be given as the variable 'output' in enDAQ cloud
     custom dashboards (to produce the dashboard)
    """
    from endaq.ide import get_doc, to_pandas
    from endaq.plot import octave_spectrogram, octave_psd_bar_plot, around_peak
    from endaq.plot.dashboards import rolling_enveloped_dashboard
    from endaq.plot.utilities import set_theme
    from endaq.cloud import create_cloud_dashboard_output

    # Set the aesthetic theme of the figures (defaults to enDAQ dark theme)
    set_theme()

    # Gets the IDE data as an idelib.dataset.Dataset object
    doc = get_doc(file_download_url)

    # A dictionary mapping channel names to pandas dataframes of that channels data
    channel_dict = {doc.channels[ch].name: to_pandas(doc.channels[ch]) for ch in doc.channels}

    # Check that the desired channel exists
    if channel not in channel_dict:
        raise ValueError(f"'{channel}' was not in the list of available channels.")

    # Check that the desired subchannel exists
    if subchannel not in channel_dict[channel]:
        raise ValueError(f"'{subchannel}' was not in the list of available subchannels of {channel}.")

    # Get the acceleration dataframe
    acceleration_df = channel_dict[channel]

    name_to_fig = dict()

    # Create a bar plot of PSD data with octave spaced frequency bins
    name_to_fig["Acceleration PSD"] = octave_psd_bar_plot(acceleration_df[[subchannel]], yaxis_title=subchannel)

    # Create a plot of the data surrounding the time of peak magnitude in the channel
    name_to_fig["Acceleration Around Peak"] = around_peak(acceleration_df, 1000)

    # Create a spectrogram with octave spaced frequency bins
    name_to_fig["Z-Axis Acceleration Spectrogram"] = octave_spectrogram(acceleration_df[[subchannel]], window=.15)[-1]

    # Generate Row of Subplots
    name_to_fig["Rolling Envelopes"] = rolling_enveloped_dashboard(
        channel_dict, plot_as_bars=True, num_rows=1, num_cols=None)

    # Create JSON Output
    output = create_cloud_dashboard_output(name_to_fig)

    return output
