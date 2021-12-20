"""
Core enDAQ Cloud communication API
"""
from __future__ import annotations

from typing import Optional, Union, Callable

from idelib.dataset import Dataset
import numpy as np
import pandas as pd
import requests
import json
import re
import urllib.request
import shutil
import os
import pathlib
import warnings


__all__ = [
    'EndaqCloud',
    'count_tags',
    'json_table_to_df',
]


ENV_PRODUCTION = "https://qvthkmtukh.execute-api.us-west-2.amazonaws.com/master"
ENV_STAGING = "https://p377cock71.execute-api.us-west-2.amazonaws.com/staging"
ENV_DEVELOP = "https://mnsz98xs64.execute-api.us-west-2.amazonaws.com/develop"


class EndaqCloud:
    """
    A representation of a connection to an enDAQ Cloud account, providing a
    high-level interface for accessing its contents.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 env: Optional[str] = None,
                 test: bool = True):
        """
        Constructor for an `EndaqCloud` object, which provides access to an
        enDAQ Cloud account.

        :param api_key: The Endaq Cloud API associated with your cloud.endaq.com account.
         If you do not have one created yet, they can be created on the following web page:
         https://cloud.endaq.com/account/api-keys
        :param env: The cloud environment to connect to, which can be production, staging, or development.
         These can be easily accessed with the variables `ENV_PRODUCTION`, `ENV_STAGING`, and `ENV_DEVELOP`
        :param test: If `True` (default), the connection to enDAQ Cloud will
            be tested before being returned. A failed test will generate a
            meaningful error message describing the problem.
        """
        self.api_key = api_key
        self.domain = env or ENV_PRODUCTION

        self.file_table = None

        self._account_id = self._account_email = None
        if test:
            info = self.get_account_info()
            if not info.get('id') or not info.get('email'):
                # TODO: change this exception; it's placeholder.
                raise RuntimeError("Failed to connect to enDAQ Cloud: response was {!r}".format(info))


    def get_account_info(self) -> dict:
        """
        Get information about the connected account. Sets or updates the
        values of `account_id` and `account_email`.

        :return: If successful, a dictionary containing (at minimum) the keys
            `email` and `id`.
        """
        response = requests.get(self.domain + "/api/v1/account/info",
                                headers={"x-api-key": self.api_key}).json()
        # Cache the ID and email. Don't clobber if the request failed
        # (just in case - it's unlikely).
        self._account_id = response.get('id', self._account_id)
        self._account_email = response.get('email', self._account_email)
        return response


    @property
    def account_id(self) -> Optional[str]:
        """ The enDAQ Cloud account's unique ID. """
        if self._account_id is None:
            self.get_account_info()
        return self._account_id


    @property
    def account_email(self) -> Optional[str]:
        """ The email address associated with the enDAQ Cloud account. """
        if self._account_email is None:
            self.get_account_info()
        return self._account_email


    def get_file(self,
                 file_id: Union[int, str],
                 local_name: Optional[str] = None) -> Dataset:
        """
        Download the specified file to local_name if provided, use the file
        name from the cloud if no local name is provided.
        
        .. todo:: This should be made to match `endaq.ide.get_doc()`

        :param file_id: The file's cloud ID.
        :param local_name: The downloaded file's destination pathname; defaults
            to the file's original basename & located in the directory in which
            the Python interpreter was launched
        :return: The imported file, as an `idelib.Dataset`.
        """
        file_url = self.domain + "/api/v1/files/download/" + file_id
        response = requests.get(file_url, headers={"x-api-key": self.api_key}).json()
        download_url = response['url']
        if local_name is None:
            local_name = response['file_name']

        with urllib.request.urlopen(download_url) as response, open(local_name, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        f = open(local_name, 'rb')

        return Dataset(f)

    def download_all_ide_files(self,
                               output_directory: Union[str, pathlib.Path] = "",
                               should_download_file_fn: Optional[Callable] = None,
                               force_reload_file_table: bool = False,
                               file_limit: int = 100) -> np.ndarray:
        """
        Downloads all IDE files from the enDAQ Cloud (up to a specified file limit).

        :param output_directory: The directory to download the ide files to
        :param should_download_file_fn: A function which accepts a row of the IDE file table and returns
         a boolean value which indicates if the IDE file should be downloaded or not.  If this function is
         not given, the default function will always return True.
        :param force_reload_file_table: If the file table to use as reference for what files exist in the cloud
         should be recomputed even if it is already stored (as `self.file_table`)
        :param file_limit: The maximum number of files to download.  If the `force_recompute_file_table` parameter
         is True then this will also be used to limit the number of files put in the file table it creates.
        :return: An array of the filenames which were just downloaded

        TO-DO:
         - Would be nice to have a parameter to get only ones with a certain tag
         - Maybe Have a blacklist and/or whitelist parameter
        """
        if not isinstance(output_directory, (str, pathlib.Path)):
            raise TypeError('the "output_directory" parameter must be given a string,'
                            f'but was given a value of type{type(output_directory)}')

        if not isinstance(force_reload_file_table, bool):
            raise TypeError('the "force_recompute_file_table" parameter must be given a value of type bool,'
                            f'but was given a value of type{type(force_reload_file_table)}')

        if not isinstance(file_limit, (int, np.integer)):
            raise TypeError('the "file_limit" parameter must be given a value of type int or np.integer,'
                            f'but was given a value of type{type(file_limit)}')

        if should_download_file_fn is None:
            should_download_file_fn = lambda x: True

        if self.file_table is None or force_reload_file_table:
            self.get_file_table(attributes=[], limit=file_limit)

        failures = []

        # The range in this iterator is just a way to force stop the loop
        for (file_name, file_data), _ in zip(list(self.file_table.iterrows()), range(file_limit)):
            if should_download_file_fn(file_data):
                try:
                    self.get_file(file_data['id'], os.path.join(output_directory, file_name))
                except:
                    failures.append(file_name)

        downloaded_filename_ary = self.file_table.index.values

        if len(failures):
            warnings.warn(f"{len(failures)} FILES FAILED TO BE DOWNLOADED!  Those files are:")
            warnings.warn(', '.join(failures))

        return downloaded_filename_ary

    def _get_files_json_response(self, limit: int = 100, attributes: Union[list, str] = "all") -> list:
        """
        A function to get attribute data about the most recent files uploaded to the cloud.

        :param limit: The maximum number of files to get info for.  The most recent files will be gotten
        :param attributes: Either a list of strings denoting the attributes to get, or a string with comma seperated
         attribute names to get.  If 'all' is specified (the default) then all attributes will be gotten
        :return: A list of the json objects
        """
        if not isinstance(limit, int):
            raise TypeError(f"the `limit` parameter must be type 'int' but type '{type(limit)}' was given.")

        if limit < 1:
            raise ValueError(f"the `limit` parameter must be at least 1, but '{limit}' was given.")

        if isinstance(attributes, str):
            attributes = attributes.split(',')
        attributes = [str(a).strip() for a in attributes]

        j = 0
        json_data = []
        while True:
            params = {
                'limit': min(100, limit - j*100),
                'attributes': attributes,
            }

            if j != 0:
                params['next_token'] = response["nextToken"]

            response = requests.get(self.domain + "/api/v1/files",
                                    params=params,
                                    headers={"x-api-key": self.api_key}).json()

            try:
                json_data += response['data']
            except KeyError:
                raise KeyError("the 'data' attribute was not present in the json response from the cloud.")

            j += 1

            if response["nextToken"] is None or j * 100 >= limit:
                break

        return json_data

    def get_file_table(self,
                       attributes: Union[list, str] = "all",
                       limit: int = 100) -> pd.DataFrame:
        """
        Get a table of the data that would be similar to that you'd get doing
        the CSV export on the my recordings page, up to the first `limit`
        files with attributes matching `attributes`.

        :param limit: The maximum number of files to return.
        :param attributes: A list of attribute strings (or a single
            comma-delimited string of attributes) to match.
        :return: A `DataFrame` of file IDs and relevant information.
        """
        json_data = self._get_files_json_response(limit=limit, attributes=attributes)

        self.file_table = json_table_to_df(json_data)

        return self.file_table

    def get_devices(self, limit: int = 100) -> pd.DataFrame:
        """
        Get dataframe of devices and associated attributes (part_number,
        description, etc.) attached to the account.

        :param limit: The maximum number of files to return.
        :return: A `DataFrame` of recorder information.
        """
        json_data = self._get_files_json_response(limit=limit, attributes=[])

        devices = {}
        for f_data in json_data:
            if f_data['device'] is not None and len(f_data['device']) > 0:
                if f_data['device']['serial_number_id'] not in devices:
                    devices[f_data['device']['serial_number_id']] = f_data['device']

        df = pd.DataFrame(devices).T

        if len(df.columns):
            df.set_index('serial_number_id')

        return df

    def set_attributes(self,
                       file_id: Union[int, str],
                       attributes: list) -> list:
        """
        Set the 'attributes' (name/value metadata) of a file.

        :param file_id: The file's cloud ID.
        :param attributes: A list of dictionaries of the following structure:

        .. code-block:: python

            [{
                "name": "attr_31",
                "type" : "float",
                "value" : 3.3,
            }]

        :return: The list of the file's new attributes.

        """
        # NOTE: This was called `post_attributes()` in the Confluence docs.
        #  'post' referred to the fact it is a POST request, which is
        #  really an internal detail; 'set' is more appropriate for an API.

        # IDEAS:
        #   * Use `**kwargs` instead of an `attributes` dict?
        #   * Automatically assume type, unless value is a tuple containing (value, type)
        for attrib in attributes:
            attrib['file_id'] = file_id

        response = requests.post(
            self.domain + "/api/v1/attributes",
            headers={"x-api-key": self.api_key},
            json={'attributes': attributes},
        )

        return response.json()




def count_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the dataframe returned by ``EndaqCloud.get_file_table()``, provide
    some info on the tags of the files in that account.

    :param df: A `DataFrame` of file information, as returned by
        ``EndaqCloud.get_file_table()``.
    :return: A `DataFrame` summarizing the tags in `df`.
    """
    # NOTE: Called `tags_count()` in Confluence docs. Function names should
    #   generally be verbs or verb phrases.
    # IDEAS:
    #   * Make this a @classmethod to make EndaqCloud the primary means of access?

    tags = {}
    for index, row in df.iterrows():
        for cur_tag in row['tags']:
            if cur_tag in tags:
                tags[cur_tag].append(index)
            else:
                tags[cur_tag] = [index]

    for tag in tags:
        tags[tag] = [len(tags[tag]), ''.join(["'", "','".join(tags[tag]), "'"])]

    return pd.DataFrame(tags, index=pd.Index(['count', 'files'], name='tag')).T



def json_table_to_df(data: list) -> pd.DataFrame:
    """
    Convert JSON parsed from a custom report to a more user-friendly
    `pandas.DataFrame`.

    :param data: A `list` of data from a custom report's JSON.
    :return: A formatted `DataFrame`
    """
    # NOTE: Steve wanted this as a separate function.
    #  Also: is this already implemented as `endaq.cloud.utilities.convert_file_data_to_dataframe()`?
    # IDEAS:
    #   * Make this a @classmethod to make EndaqCloud class and/or instances the primary means of access?
    df = pd.DataFrame(data)
    df['attributes'] = df['attributes'].map(lambda x: {attribs['name']: attribs for attribs in x})

    unique_attributes_and_types = df['attributes'].map(lambda x: [(k, v['type']) for k, v in x.items()]).values
    unique_attributes_and_types = set(pair for file_info in unique_attributes_and_types for pair in file_info)

    for attrib_name, attrib_type_str in unique_attributes_and_types:
        if attrib_type_str == 'float':
            df[attrib_name] = df['attributes'].map(
                lambda x: None if len(x) == 0 or attrib_name not in x else float(x[attrib_name]['value']))
        elif attrib_type_str == 'string':
            try:  # Try and parse the JSON string into an array of floats
                df[attrib_name] = df['attributes'].map(
                    lambda x: [] if len(x) == 0 or attrib_name not in x else np.array(
                        json.loads(re.sub(r'\bnan\b', 'NaN', x[attrib_name]['value'])), dtype=np.float32))
            except json.JSONDecodeError:  # Save it as a String if it can't be converted to a float array
                df[attrib_name] = df['attributes'].map(
                    lambda x: "" if len(x) == 0 or attrib_name not in x else x[attrib_name]['value'])

    # Convert the columns which represent times to pandas datetime type
    for time_col_name in ['recording_ts', 'created_ts', 'modified_ts', 'archived_ts']:
        if time_col_name in df:
            df[time_col_name] = pd.to_datetime(df[time_col_name], unit='s')

    if 'gpsLocationFull' in df:
        # Add the GPS coordinates as 2 seperate latitude and longitude columns
        gps_coord_series = df['gpsLocationFull'].map(
            lambda x: np.array(x.split(','), dtype=np.float32) if len(x) else np.array(2 * [np.nan],
                                                                                       dtype=np.float32))
        df['latitudes'] = gps_coord_series.map(lambda x: x[0])
        df['longitudes'] = gps_coord_series.map(lambda x: x[1])

    df = df.set_index('file_name')

    return df
