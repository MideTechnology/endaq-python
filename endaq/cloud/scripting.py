import requests
from types import FunctionType
import plotly.io as pio
import json

from .core import EndaqCloud, ENV_PRODUCTION, ENV_STAGING, ENV_DEVELOP


__all__ = [
    'create_cloud_dashboard_output',
    'produce_dashboard_plots',
]


def create_cloud_dashboard_output(name_to_fig: dict) -> str:
    """
    A function which makes producing the json based string used to produce custom enDAQ Cloud report
    dashboards easy.

    :param name_to_fig: A dictionary mapping the desired names/titles of plots to a Plotly figure.
     The dictionary must have 4 elements, and the ordering DOES matter as it dictates the order
     the plots will be added to the dashboard.  This means the ordering of Python dictionaries which
     appears in Python 3.7 is crucial for using this!
    :return: The json based string which is to be given as the variable 'output' in enDAQ cloud
     custom dashboards
    """
    if not isinstance(name_to_fig, dict):
        raise TypeError(f"'name_to_fig' parameter must be a dictionary, but was given {type(name_to_fig)} instead")

    if len(name_to_fig) != 4:
        raise ValueError("The number of (key, value) pairs in 'name_to_fig' must be exactly 4, "
                         f"but {len(name_to_fig)} are given")

    return "[" + ", ".join([v.to_json()[:-1] + ', "title": "' + k + '"}' for k, v in name_to_fig.items()]) + "]"


def produce_dashboard_plots(dashboard_script_fn: FunctionType, api_key: str, max_num_files: int = 100,
                            environment: str = 'production', display_plots: bool = True) -> list:
    """
    A function used to simulate a run of a desired enDAQ Cloud custom report script without needing to use
    cloud.endaq.com

    :param dashboard_script_fn: A function accepting the parameters `files` and `file_download_url`, which has all
     the exact code that would be put into a enDAQ Cloud custom report script, followed by one final line:
     `return output`
    :param api_key: The enDAQ Cloud API key
    :param max_num_files: The maximum number of files to get data about.  Specifically, this is used to
     specify how many of the most recently uploaded IDE files in the cloud will have their info passed to
     your custom report script (through a list of json blobs, as parameter 'files')
    :param environment: The version of the enDAQ Cloud to communicate with, the options are 'production', 'staging',
     or 'develop'.  This should only be used internally at Mide
    :param display_plots: If the plots being produced should be displayed
    :return: A list of the 4 plotly figures produced
    """
    if environment == 'production':
        api_access_url = ENV_PRODUCTION
    elif environment == 'staging':
        api_access_url = ENV_STAGING
    elif environment == 'develop':
        api_access_url = ENV_DEVELOP
    else:
        raise ValueError("Only 'production', 'staging', and 'develop' may be given for the 'environment' parameter, "
                         f" but {environment} was given instead.")

    parameters = {"x-api-key": api_key}

    cloud_obj = EndaqCloud(api_key, env=api_access_url)

    files = cloud_obj._get_files_json_response(limit=max_num_files)

    most_recent_file_id = files[-1]['id']

    file_download_url = requests.get(
        api_access_url + '/api/v1/files/download/' + most_recent_file_id,
        headers=parameters
    ).json()['url']

    output = dashboard_script_fn(files=files, file_download_url=file_download_url)

    figures = [pio.from_json(json.dumps(blob)) for blob in json.loads(output)]

    if display_plots:
        for fig in figures:
            fig.show()

    return figures
