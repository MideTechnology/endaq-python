"""
gdrive.py: Accessing data via Google Drive links.
"""
import re
from urllib.parse import parse_qs, urlparse

import requests

# DRIVE_URL = "https://docs.google.com/uc?export=download"
DRIVE_URL = "https://docs.google.com/uc"


def get_file_id(url):
    """
    Extract the Google Drive file ID from a URL. The ID can be in the
    URL path itself, or a query argument.

    :param url: The Google Drive 'shared link'.
    :return: The file ID portion of the URL.
    """
    # NOTE: This is a somewhat brittle hack. Revised later?
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    qid = query.get('id', '')
    if qid:
        return qid[0]

    match = re.match(r"^/file/d/(.*?)/view$", parsed.path)
    if match:
        return match.groups()[0]

    return None


def gdrive_download(url, localfile, params=None, cookies=None, drive_url=DRIVE_URL):
    """
    Retrieve an IDE from Google Drive. The file must be set to be shared
    with anyone with the URL.

    :param url: The 'shared link' to the file on Google Drive.
    :param localfile: The local filename (if saving the file).
    :param params: Additional (optional) request parameters.
    :param cookies: Optional browser cookies for the session.
    :param drive_url: The Google Docs download URL.
    :return: The 'get' response and the filename.
    """
    file_id = get_file_id(url)
    if not file_id:
        raise ValueError(f"Could not identify ID in URL {url}")

    p = {'id': file_id}
    if params:
        p.update(params)

    session = requests.Session()
    response = session.get(drive_url, params=p, cookies=cookies, stream=True)

    if not response.ok:
        raise ValueError(f"Could not retrieve data from URL {url} "
                         f"({response.status_code}: {response.reason})")

    # A response with the file will have its name in the headers.
    name = None

    if 'Content-Disposition' not in response.headers:
        # TODO: This is probably an intermediate page, possibly a warning.
        #  Get the download URL from the response body text and/or
        #  authorization token from the cookies, then get another response.
        raise ValueError(f"Could not retrieve data from URL {url} "
                         f"(not shared 'anyone with link'?)")

    if 'Content-Disposition' in response.headers:
        m = re.search('filename="(.*)"', ''.join(response.headers['Content-Disposition']))
        if m:
            name = m.groups()[0]

    if not name:
        raise ValueError(f"Could not retrieve data from URL {url}")

    return response, name
