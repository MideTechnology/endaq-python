from __future__ import annotations

import plotly.io as pio
import plotly.graph_objects as go
import numpy as np
import typing
from typing import Union


def define_theme(template_name: str = "endaq_cloud", default_plotly_template: str = 'plotly_dark',
                 text_color: str = '#DAD9D8', font_family: str = "Open Sans",
                 title_font_family: str = "Open Sans SemiBold", graph_line_color: str = '#DAD9D8',
                 grid_line_color: str = "#404041", background_color: str = '#262626',
                 plot_background_color: str = '#0F0F0F') -> go.layout._template.Template:
    """
    Define a Plotly theme (template), allowing completely custom aesthetics

    :param template_name: The name for the Plotly template being created
    :param default_plotly_template: The default Plotly Template (aspects of this will be used if
     a characteristic isn't set elsewhere)
    :param text_color: The color of the text
    :param font_family: The font family to use for text (not including the title)
    :param title_font_family: The font family to use for the title
    :param graph_line_color: The line color used when plotting line plots
    :param grid_line_color: The color of the grid lines on the plot
    :param background_color: The background color of the figure
    :param plot_background_color: The background color of the plot
    :return: The plotly template which was just created
    """
    pio.templates[template_name] = pio.templates[default_plotly_template]

    # Line Colors
    colorway = ['#EE7F27', '#6914F0', '#2DB473', '#D72D2D', '#3764FF', '#FAC85F', '#27eec0', '#b42d4d', '#82d72d',
                '#e35ffa']
    colorbar = [[0.0, '#6914F0'],
                [0.2, '#3764FF'],
                [0.4, '#2DB473'],
                [0.6, '#FAC85F'],
                [0.8, '#EE7F27'],
                [1.0, '#D72D2D']]
    pio.templates[template_name]['layout']['colorway'] = colorway
    pio.templates[template_name]['layout']['colorscale']['sequential'] = colorbar
    pio.templates[template_name]['layout']['colorscale']['sequentialminus'] = colorbar
    pio.templates[template_name]['layout']['colorscale']['diverging'] = [[0.0, '#6914F0'],
                                                                         [0.5, '#f7f7f7'],
                                                                         [1.0, '#EE7F27']]
    plot_types = ['contour', 'heatmap', 'heatmapgl', 'histogram2d', 'histogram2dcontour', 'surface']
    for p in plot_types:
        pio.templates[template_name]['data'][p][0].colorscale = colorbar

    # Text
    # dictionary = dict(font=dict(family="Open Sans", size=24, color=text_color))
    # pio.templates[template_name]['layout']['annotations'] = [(k, v) for k, v in dictionary.items()]
    pio.templates[template_name]['layout']['font'] = dict(family=font_family, size=16, color=text_color)
    pio.templates[template_name]['layout']['title_font_family'] = title_font_family
    pio.templates[template_name]['layout']['title_font_size'] = 24
    pio.templates[template_name]['layout']['title_x'] = 0.5
    pio.templates[template_name]['layout']['yaxis_title_font_size'] = 20
    pio.templates[template_name]['layout']['xaxis_title_font_size'] = 20

    # Legend
    pio.templates[template_name]['layout']['legend'] = dict(orientation='h', y=-0.2)

    # Background Color
    pio.templates[template_name]['layout']['paper_bgcolor'] = background_color
    pio.templates[template_name]['layout']['plot_bgcolor'] = plot_background_color
    pio.templates[template_name]['layout']['geo']['bgcolor'] = plot_background_color
    pio.templates[template_name]['layout']['polar']['bgcolor'] = plot_background_color
    pio.templates[template_name]['layout']['ternary']['bgcolor'] = plot_background_color
    pio.templates[template_name]['layout']['scene']['xaxis']['backgroundcolor'] = plot_background_color
    pio.templates[template_name]['layout']['scene']['yaxis']['backgroundcolor'] = plot_background_color
    pio.templates[template_name]['layout']['scene']['zaxis']['backgroundcolor'] = plot_background_color

    # Graph Lines
    pio.templates[template_name]['data']['scatter'][0].marker.line.color = graph_line_color
    pio.templates[template_name]['layout']['scene']['xaxis']['gridcolor'] = grid_line_color
    pio.templates[template_name]['layout']['scene']['xaxis']['linecolor'] = graph_line_color
    pio.templates[template_name]['layout']['scene']['yaxis']['gridcolor'] = grid_line_color
    pio.templates[template_name]['layout']['scene']['yaxis']['linecolor'] = graph_line_color
    pio.templates[template_name]['layout']['scene']['zaxis']['gridcolor'] = grid_line_color
    pio.templates[template_name]['layout']['scene']['zaxis']['linecolor'] = graph_line_color
    pio.templates[template_name]['layout']['xaxis']['gridcolor'] = grid_line_color
    pio.templates[template_name]['layout']['xaxis']['linecolor'] = graph_line_color
    pio.templates[template_name]['layout']['yaxis']['gridcolor'] = grid_line_color
    pio.templates[template_name]['layout']['yaxis']['linecolor'] = graph_line_color
    pio.templates[template_name]['layout']['yaxis']['zerolinecolor'] = graph_line_color
    pio.templates[template_name]['layout']['xaxis']['zerolinecolor'] = graph_line_color
    pio.templates[template_name]['layout']['yaxis']['zerolinewidth'] = 1
    pio.templates[template_name]['layout']['xaxis']['zerolinewidth'] = 1
    pio.templates[template_name]['layout']['yaxis']['showline'] = True
    pio.templates[template_name]['layout']['xaxis']['showline'] = True
    pio.templates[template_name]['layout']['yaxis']['showgrid'] = True
    pio.templates[template_name]['layout']['xaxis']['showgrid'] = True
    pio.templates[template_name]['layout']['yaxis']['mirror'] = True
    pio.templates[template_name]['layout']['xaxis']['mirror'] = True
    pio.templates[template_name]['layout']['yaxis']['zeroline'] = True
    pio.templates[template_name]['layout']['xaxis']['zeroline'] = True

    return pio.templates[template_name]


def set_theme(theme: typing.Literal['endaq', 'endaq_light', 'endaq_arial', 'endaq_light_arial'] = 'endaq',
              ) -> go.layout._template.Template:
    """
    Set the plot appearances based on a known 'theme'.

    :param theme: A string denoting which plot appearance color scheme to use.
     Current options are `'endaq'`, `'endaq_light'`, `'endaq_arial'` and `'endaq_light_arial'`.
    :return: The plotly template which was set
    """
    if not isinstance(theme, str):
        raise TypeError("'theme' must be given as a string")
        
    theme = theme.lower()
    
    if theme not in ['endaq_cloud', 'endaq_cloud_light', 'endaq', 'endaq_light']:
        raise ValueError("'" + theme + "' not an option")

    define_theme()

    define_theme(template_name='endaq', font_family="Arial", title_font_family="Arial")

    define_theme(template_name='endaq_cloud_light', grid_line_color='#DAD9D8', graph_line_color='#404041',
                 plot_background_color='#f3f3f3', background_color='#FFFFFF', text_color='#404041',
                 default_plotly_template='plotly_white')

    define_theme(template_name='endaq_light', grid_line_color='#DAD9D8', graph_line_color='#404041',
                 plot_background_color='#f3f3f3', background_color='#FFFFFF', text_color='#404041',
                 font_family="Arial", title_font_family="Arial",
                 default_plotly_template='plotly_white')

    # Set Default
    pio.templates.default = theme

    return pio.templates[theme]


def get_center_of_coordinates(lats: np.ndarray, lons: np.ndarray, as_list: bool = False, as_degrees: bool = True
                              ) -> Union[list, dict]:
    """
    Inputs and outputs are measured in degrees.
    
    :param lats: An ndarray of latitude points
    :param lons: An ndarray of longitude points
    :param as_list: If True, return a length 2 list of the latitude and longitude coordinates.   If not return a
     dictionary of format {"lon": lon_center, "lat": lat_center}
    :param as_degrees: A boolean value representing if the 'lats' and 'lons' parameters are given in degrees (as opposed
     to radians).  These units will be used for the returned values as well.  
    :return: The latitude and longitude values as either a dictionary or a list, which is
     determined by the value of the `as_list` parameter (see the `as_list` docstring for details
     on the formatting of this return value
    """
    # Convert coordinates to radians if given in degrees
    if as_degrees:
        lats *= np.pi / 180
        lons *= np.pi / 180

    # Convert coordinates to 3D coordinates
    x_coords = np.cos(lats) * np.cos(lons)
    y_coords = np.sin(lats) * np.cos(lons)
    z_coords = np.sin(lons)

    # Caluculate the means of the coordinates in 3D
    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    z_mean = np.mean(z_coords)

    # Convert back to lat/lon from 3D coordinates
    lat_center = np.arctan2(y_mean, x_mean)
    lon_center = np.arctan2(z_mean, np.sqrt(x_mean ** 2 + y_mean ** 2))

    # Convert back to degrees from radians
    if as_degrees:
        lat_center *= 180 / np.pi
        lon_center *= 180 / np.pi

    if as_list:
        return [lat_center, lon_center]

    return {
        "lat": lat_center,
        "lon": lon_center,
    }



def determine_plotly_map_zoom(
        lons: tuple = None,
        lats: tuple = None,
        lonlats: tuple = None,
        projection: str = "mercator",
        width_to_height: float = 2.0,
        margin: float = 1.2,
) -> float:
    """
    Finds optimal zoom for a plotly mapbox. Must be passed (lons & lats) or lonlats.

    Originally based on the following post:
    https://stackoverflow.com/questions/63787612/plotly-automatic-zooming-for-mapbox-maps
    
    This is a temporary solution awaiting an official implementation:
    https://github.com/plotly/plotly.js/issues/3434
    
    :param lons: tuple, optional, longitude component of each location
    :param lats: tuple, optional, latitude component of each location
    :param lonlats: tuple, optional, gps locations
    :param projection: str, only accepting 'mercator' at the moment,
        raises `NotImplementedError` if other is passed
    :param width_to_height: float, expected ratio of final graph's with to height,
        used to select the constrained axis.
    :param margin: The desired margin around the plotted points (where 1 would be no-margin)
    :return: The zoom scaling for the Plotly map
    
    .. note::
      This implementation could be potentially problematic.  By simply averaging min/max coorindates
      you end up with situations such as the longitude lines -179.99 and 179.99 being
      almost right next to each other, but their center is calculated at 0, the other side of the earth.
    """
    if lons is None and lats is None:
        if isinstance(lonlats, tuple):
            lons, lats = zip(*lonlats)
        else:
            raise ValueError("Must pass lons & lats or lonlats")
            
    # longitudinal range by zoom level (20 to 1) in degrees, log scaled, with 360 as min zoom
    lon_zoom_range = np.array([360 / 2 ** j for j in range(20)[::-1]], dtype=np.float32)

    if projection == "mercator":
        maxlon, minlon = max(lons), min(lons)
        maxlat, minlat = max(lats), min(lats)
        
        height = (maxlat - minlat) * margin * width_to_height
        width = (maxlon - minlon) * margin
        lon_zoom = np.interp(width, lon_zoom_range, range(20, 0, -1))
        lat_zoom = np.interp(height, lon_zoom_range, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
    else:
        raise NotImplementedError(f"{projection} projection is not implemented")

    return zoom
    
