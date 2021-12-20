def cloud_correlation_script(files: list, file_download_url: str):
    """
    Creates a cloud dashboard (for use in cloud.endaq.com custom report generation) for comparing various recording
    files and exploring the correlations between them.  It creates a PCA plot with dropdown menus to select the
    PCA components, a plot which compares different file attribute values against eachother, a t-SNE plot, and
    all the attribute values.

    :param files: A `list` of JSON blobs of recording file data as it's given in a custom report in cloud.endaq.com
    :param file_download_url: The download URL for the IDE file most recently uploaded to cloud.endaq.com
    :return: The json based string which is to be given as the variable 'output' in enDAQ cloud
     custom dashboards (to produce the dashboard)
    """
    import numpy as np
    from endaq.plot.utilities import set_theme
    from endaq.cloud.core import json_table_to_df
    from endaq.plot import get_tsne_plot, get_pure_numpy_2d_pca, general_get_correlation_figure,\
        multi_file_plot_attributes
    from endaq.cloud import create_cloud_dashboard_output

    # Set the aesthetic theme of the figures, and store the list of colors used as a colorway for future use
    colorway = set_theme()['layout']['colorway']

    # Remove all file data which doesn't contain any attribute data
    files = [f for f in files if f['attributes']]

    # Ensure that there are at least 2 files
    if len(files) <= 2:
        raise ValueError(f"Must have at least 2 files given to analyze, was given {len(files)} instead")

    # Get the dataframe of the file attribute data
    df = json_table_to_df(files)

    # Create a dictionary mapping serial number of the enDAQ recording device with
    unique_serial_nums = np.unique(df['serial_number_id'])
    serials_to_colors = {serial_num: c for c, serial_num in
                        zip(colorway * (1 + len(unique_serial_nums) // len(colorway)), unique_serial_nums)}

    # The dictionary to be used to create the desired output of this function from (see the final line of this script
    # for how it's used to accomplish this)
    name_to_fig = dict()

    # Create PCA plot with 2 drop-down menus
    name_to_fig["PCA Plot"] = get_pure_numpy_2d_pca(df, color_discrete_map=serials_to_colors)

    # Create a plot that plots one attribute value against another attribute value, the two
    # attributes are selected via two dropdown menus
    name_to_fig["Attribute Correlation Explorer"] = general_get_correlation_figure(
        df,
        color_discrete_map=serials_to_colors,
        characteristics_to_show_on_hover=['id', 'recording_length', 'accelerationRMSFull', 'serial_number_id'],
    )

    # Create the t-SNE visualization
    name_to_fig["t-SNE"] = get_tsne_plot(df, color_discrete_map=serials_to_colors)

    # Generate Row of Subplots
    name_to_fig["Measurements Pane"] = multi_file_plot_attributes(
        df, recording_colors=df['serial_number_id'].map(serials_to_colors.get))

    # Create JSON Output
    output = create_cloud_dashboard_output(name_to_fig)

    return output
