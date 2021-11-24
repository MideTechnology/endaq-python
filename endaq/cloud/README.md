# endaq-cloud

The `cloud` sub-module for `endaq-python`

This module houses tools and functions to make both interacting with the enDAQ Cloud API and the [enDAQ Cloud](https://cloud.endaq.com) itself easier.

## Tools:

### API Wrapper:

The API Wrapper provides a simple command-line interface for accessing basic file and device information from the enDAQ Cloud API. Output of all commands except `account` and `attributes` are in `csv` files in the `output` folder.

To access the cloud, this tool requires an API key associated with a user's enDAQ Cloud account, which can be provided in two ways:

* (recommended) add to the `endaq.cloud` project directory a `.env` file, formatted like so:

		API_KEY=<Your Key>

* pass in an API key through the command line using the `--key` option

#### warning
For security reasons, it is generally discouraged to make an authentication key visible on-screen or accessible through the clipboard, such as when using the `--key` option; we provide the `key` option solely as a convenience.

#### Commands

- files               Outputs file information for selected number of files and attributes
- file-id             Outputs file information for file with specified ID to output file
- devices             Ouputs device information for selected number of files
- device-id           Output device information for device with specified ID to output file
- account             Prints out account information
- attributes          Adds an attribute to a specified file
- set-env             Creates a .env file with passed in API key *NOT SECURE*

#### Usages

Flag | Description 
---|---
-h                  | Command Line Help
--id, -i            | File or Device ID
--limit, -l         | File or Device output limit; Max 100 default 50
--key, -k           | API Key
--attributes, -a    | Attributes to be outputted; options = all or att1,att2...; default is none
--name, -n          | Attribute Name
--type, -t          | Attribute Type; options = int, float, string, boolean
--value, -v         | Attribute Value
--verbose, -V       | Prints out URL API calls


- endaq-cloud set-env -k <API_KEY>
- endaq-cloud download -i <FILE_ID> -o <OUTPUT_FOLDER>
- endaq-cloud files -a <ATTRIBUTES_TO_GET> -l <FILE_OR_DEVICE_OUTPUT_LIMIT>
- endaq-cloud file-id -i **# additional argument needed**
- endaq-cloud devices -l <FILE_OR_DEVICE_OUTPUT_LIMIT>
- endaq-cloud device-id -i **# additional argument needed**
- endaq-cloud account
- endaq-cloud attribute -n <ATTRIBUTE_NAME> -t <ATTRIBUTE_TYPE> -v <ATTRIBUTE_VALUE> -i **# additional arguments needed**



### Utilities:

Convienience functions for interacting with the cloud API and the enDAQ custom cloud report generation.
