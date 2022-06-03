import os.path
import random
import time
import unittest

from idelib.dataset import Dataset
from idelib.importer import importFile
from endaq.ide import files


IDE_FILENAME = os.path.join(os.path.dirname(__file__), "test.ide")
IDE_LOCAL_URL = "file://" + IDE_FILENAME.replace('\\', '/')

IDE_URL_HTTP = "http://mide.services/software/test.ide"
IDE_URL_HTTPS = "https://mide.services/software/test.ide"
IDE_GDRIVE_URL = "https://drive.google.com/file/d/1t3JqbZGhuZbIK9agH24YZIdVE26-NOF5/view?usp=sharing"


class GetDocTests(unittest.TestCase):

    def setUp(self):
        self.dataset = importFile(IDE_FILENAME)


    def test_get_doc_basics(self):
        # Parameter tests
        self.assertRaises(TypeError, files.get_doc)
        self.assertRaises(TypeError, files.get_doc, IDE_FILENAME, url=IDE_LOCAL_URL)
        self.assertRaises(TypeError, files.get_doc, filename=IDE_FILENAME, url=IDE_LOCAL_URL)


    def test_get_doc_files(self):
        """ Test basic opening of a file from a filename. """
        # The baseline file
        doc1 = files.get_doc(IDE_FILENAME)
        self.assertEqual(len(self.dataset.ebmldoc), len(doc1.ebmldoc),
                         "get_doc(filename) copy length did not match that of local copy")
        self.assertTrue(all(a == b for a, b in zip(self.dataset.ebmldoc, doc1.ebmldoc)),
                        "get_doc(filename) copy contents did not match that of local copy")

        self.assertIsInstance(doc1, Dataset,
                              "get_doc(filename) did not return a Dataset")

        # Verify `name` and `filename` parameters equivalent
        doc2 = files.get_doc(filename=IDE_FILENAME)
        self.assertEqual(doc1.filename, doc2.filename)

        # Verify a "file:" URL (with forward slashes) translates
        doc3 = files.get_doc(IDE_LOCAL_URL)
        self.assertEqual(doc1.filename, doc3.filename)

        # Verify a "file:" URL (with Windows delimiter) translates
        if "\\" in IDE_FILENAME:
            doc4 = files.get_doc("file://" + IDE_FILENAME)
            self.assertEqual(doc1.filename, doc4.filename)

        # Verify validation using known non-IDE (this file)
        self.assertRaises(ValueError, files.get_doc, __file__)


    def test_get_doc_url(self):
        """ Test getting an IDE from a URL. """
        # This is admittedly a simplistic test, but it reveals a great deal.
        # It is unlikely that only the content within EBML elements would be
        # corrupted but not the EBML structure.
        doc_http = files.get_doc(IDE_URL_HTTP)
        self.assertIsInstance(doc_http, Dataset,
                              "get_doc() did not return a Dataset")
        self.assertEqual(len(self.dataset.ebmldoc), len(doc_http.ebmldoc),
                         "HTTP copy length did not match that of local copy")
        self.assertTrue(all(a == b for a, b in zip(self.dataset.ebmldoc, doc_http.ebmldoc)),
                        "HTTP copy contents did not match that of local copy")

        doc_https = files.get_doc(IDE_URL_HTTPS)
        self.assertIsInstance(doc_https, Dataset,
                              "get_doc() did not return a Dataset")
        self.assertEqual(len(self.dataset.ebmldoc), len(doc_https.ebmldoc),
                         "HTTPS copy length did not match that of local copy")
        self.assertTrue(all(a == b for a, b in zip(self.dataset.ebmldoc, doc_https.ebmldoc)),
                        "HTTPS copy contents did not match that of local copy")


    def test_get_doc_gdrive(self):
        """ Test getting an IDE from a Google Drive URL. """
        # There are sporadic HTTP failures getting the sample file from
        # Google Drive, possibly a result of the GitHub tests suspiciously
        # hammering the URL. This is a naive implementation allowing 3
        # retries, with a random sleep in between.
        retries = 2
        doc = None
        while retries:
            try:
                doc = files.get_doc(IDE_GDRIVE_URL)
                break
            except ValueError as err:
                retries -= 1
                if "403: Forbidden" not in str(err) or not retries:
                    raise
                else:
                    time.sleep(random.random() * 3)

        self.assertIsInstance(doc, Dataset,
                              "get_doc() did not return a Dataset")
        self.assertEqual(len(self.dataset.ebmldoc), len(doc.ebmldoc),
                         "Google Drive copy length did not match that of local copy")
        self.assertTrue(all(a == b for a, b in zip(self.dataset.ebmldoc, doc.ebmldoc)),
                        "Google Drive copy contents did not match that of local copy")


if __name__ == '__main__':
    unittest.main()
