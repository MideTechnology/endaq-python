import argparse
import pathlib
import sys
import textwrap
import urllib.parse

import requests
import os
from dotenv import load_dotenv, find_dotenv
import csv

load_dotenv(find_dotenv())

URL = 'https://qvthkmtukh.execute-api.us-west-2.amazonaws.com/master'
PARAMETERS = {
        "x-api-key": os.environ.get('API_KEY')
    }


def attributes_file(data, output_path):
    """
    Write the given attributes to a file.

    :param data: All the attributes for the files
    :param output_path: File path
    """
    with open(output_path + 'attributes.csv', 'w', newline='') as att_file:
        csv_writer = csv.writer(att_file)
        csv_writer.writerow(['file_id', 'attribute'])
        for x in data:
            for y in x['attributes']:
                csv_writer.writerow([x['id'], y])


def account_info(verbose):
    """Request account info from the cloud and print the results."""
    if verbose:
        print('GET', URL + '/api/v1/account/info')
    response = requests.get(URL + '/api/v1/account/info', headers=PARAMETERS)
    print(response.json())


def all_files(att, limit, output_path, verbose):
    """
    Request file data from the cloud and write the output to a file.

    :param att: list of attributes to be shown
    :param limit: how many files to be listed
    :param output_path: location of output files
    """
    query_data = {k: v for k, v in dict(limit=limit, attributes=att).items() if v}
    query = ('?' + urllib.parse.urlencode(query_data)) if query_data else ''

    if verbose:
        print('GET', URL + '/api/v1/files' + query)
    response = requests.get(URL + '/api/v1/files' + query, headers=PARAMETERS)
    data = response.json()['data']
    if att:
        attributes_file(data, output_path)

    with open(output_path + 'files.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        headers = list(data[0].keys())
        headers.remove('attributes')
        csv_writer.writerow(headers)
        for x in data:
            del x['attributes']
            csv_writer.writerow(x.values())


def download_file(file_id, output_path=None):
    """Download the file with the given ID."""
    # Request a download link for the file
    request_url = f"{URL}/api/v1/files/download/{file_id}"
    response = requests.get(request_url, headers=PARAMETERS)
    try:
        response.raise_for_status()
    except requests.HTTPError as ex:
        raise RuntimeError(f"failed to retrieve file download link ({response.status_code})")

    # Request the contents of the file
    response_json = response.json()
    download_url = response_json["url"]
    download_filename = response_json["file_name"]
    download_response = requests.get(download_url, stream=True)
    try:
        download_response.raise_for_status()
    except requests.HTTPError as ex:
        raise RuntimeError(f"failed to download file ({download_response.status_code})")

    # Save the contents of the file to disk
    output_path = pathlib.Path(output_path or ".")
    file_path = output_path / download_filename
    with open(file_path, "wb") as file:
        for chunk in download_response.iter_content(chunk_size=None):
            file.write(chunk)


def file_by_id(id_, output_path, verbose):
    """
    Request a specific file's data from the cloud and write the output to files.

    :param id_: File ID
    :param output_path: Output directory
    """
    if verbose:
        print('GET', URL + '/api/v1/files/' + id_)
    response = requests.get(URL + '/api/v1/files/' + id_, headers=PARAMETERS)
    data = response.json()
    if 'code' in data:
        raise ValueError(f'invalid file ID "{id_}"')

    with open(output_path + 'attributes.csv', 'w', newline='') as att_file:
        csv_writer = csv.writer(att_file)
        headers = list(data['attributes'][0].keys())
        csv_writer.writerow(headers)
        for x in data['attributes']:
            csv_writer.writerow(x.values())

    del data['attributes']
    with open(output_path + 'file_' + id_ + '.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        headers = data.keys()
        csv_writer.writerow(headers)
        csv_writer.writerow(data.values())


def devices(output_path, limit, verbose):
    """
    Request devices' data from the cloud and write the output to a file.

    :param output_path: output file location
    :param limit: number of devices shown
    """
    if verbose:
        print('GET', URL + '/api/v1/devices/')
    if limit == '':
        response = requests.get(URL + '/api/v1/devices/', headers=PARAMETERS)
    else:
        response = requests.get(URL + '/api/v1/devices?limit=' + limit, headers=PARAMETERS)

    data = response.json()['data']
    with open(output_path + 'devices.csv', 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file)
        headers = list(data[0].keys())
        csv_writer.writerow(headers)
        for x in data:
            csv_writer.writerow(x.values())


def device_by_id(id_, output_path, verbose):
    """
    Request a specific device's data from the cloud and write the output to a file.
    
    :param id_: ID of specific device
    :param output_path: output location for information
    """
    if verbose:
        print('GET', URL + '/api/v1/devices/' + id_)
    response = requests.get(URL + '/api/v1/devices/' + id_, headers=PARAMETERS)

    data = response.json()
    if 'code' in data:
        raise ValueError(f'invalid device ID "{id_}"')
    data_file = open(output_path + 'device_' + id_ + '.csv', 'w', newline='')
    csv_writer = csv.writer(data_file)
    headers = data.keys()
    csv_writer.writerow(headers)
    csv_writer.writerow(data.values())


def post_attribute(id_, name, type_, value, verbose):
    """
    Post attribute data to the cloud and print the results.

    :param id_: file ID
    :param name: attribute name
    :param type_: attribute type
    :param value: attribute value
    """
    if verbose:
        print('POST', URL + '/api/v1/attributes')
    attributes = {"name": name,
                  "type": type_,
                  "value": value,
                  "file_id": id_}
    att = requests.post(URL + '/api/v1/attributes', headers=PARAMETERS, json={"attributes": [attributes]})
    print(att.json())


def make_env(key):
    with open('.env', 'w') as f:
        f.write(f'API_KEY={key}')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    Commands:
        - files               Outputs file information for selected number of files and attributes
        - file-id             Outputs file information for file with specified ID to output file
        - devices             Ouputs device information for selected number of files
        - device-id           Output device information for device with specified ID to output file
        - account             Prints out account information
        - attributes          Adds an attribute to a specified file
        - set-env             Creates a .env file with passed in API key *NOT SECURE*
    '''))
    parser.add_argument('command', choices=['files', 'download', 'file-id', 'devices', 'device-id', 'account', 'attribute',
                                            'set-env'])
    parser.add_argument('--id', '-i', default='', help='Device or File id')
    parser.add_argument('--attributes', '-a', default='', help='What attributes you want to view; default is none, '
                                                               'can be all, or att1,att2,...')
    parser.add_argument('--limit', '-l', default='', help='How many devices or files you want returned; Max 100')
    parser.add_argument('--key', '-k', default='', help='API key if it is not in the .env file or you want to set env')
    parser.add_argument('--name', '-n', help='Name of new attribute')
    parser.add_argument('--type', '-t', help='Type of element the new attribute is')
    parser.add_argument('--value', '-v', help='Value of new attribute')
    parser.add_argument('--verbose', '-V', action='store_true', help='Shows API url calls')

    parser.add_argument('--url', '-u', default='', help=argparse.SUPPRESS)
    parser.add_argument('--output', '-o', default='')
    args = parser.parse_args()

    # Specifies output directory if one is passed in
    output = args.output or './output/'
    if not os.path.exists(output):
        os.makedirs(output)

    # changes API url if one is passed in. (ONLY FOR DEVS)
    global URL
    if args.url != '':
        URL = args.url

    # changes API key if one is passed in
    if args.key != '':
        PARAMETERS.update({"x-api-key": args.key})
    elif os.environ.get('API_KEY') is None:
        sys.exit('Create a .env with API Key or pass it in with --key')

    if args.command == 'account':
        account_info(args.verbose)
    elif args.command == 'download':
        download_file(args.id, output)
    elif args.command == 'files':
        all_files(args.attributes, args.limit, output, args.verbose)
        print('output can be found in output/files.csv and output/attributes.csv')
    elif args.command == 'file-id' and args.id != '':
        file_by_id(args.id, output, args.verbose)
        print(f'output can be found in output/file_{args.id}.csv and output/attributes.csv')
    elif args.command == 'devices':
        devices(output, args.limit, args.verbose)
        print('output can be found in devices.csv')
    elif args.command == 'device-id' and args.id != '':
        device_by_id(args.id, output, args.verbose)
        print(f'output can be found in device_{args.id}.csv')
    elif args.command == 'attribute':
        if args.name is not None and args.type is not None and args.value is not None and args.id is not None:
            post_attribute(args.id, args.name, args.type, args.value, args.verbose)
        else:
            sys.exit('id, name, type, and value can\'t be empty')
    elif args.command == 'set-env':
        if args.key == '':
            sys.exit('Need -k or --key for this command')
        else:
            make_env(args.key)
    else:
        sys.exit('Use -h for help with commands.')


if __name__ == '__main__':
    main()
