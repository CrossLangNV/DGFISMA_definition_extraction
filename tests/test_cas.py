import base64
import json
import os
import unittest

import requests

"""
Make sure Update to your localhost URL
"""
URL = 'http://localhost:10001/'

URL_extract_defs = URL + 'extract_definitions'

ROOT = os.path.dirname(__file__)


class TestApp(unittest.TestCase):
    def test_home(self):
        x = requests.get(URL)

        self.assertEqual(200, x.status_code)


class TestCas(unittest.TestCase):
    def test_post_status(self):
        x = requests.post(URL_extract_defs,
                          # json=data
                          )

        self.assertEqual(400, x.status_code, "Can't work if no data is provided")

        data = {"cas_content": ""}
        x = requests.post(URL_extract_defs,
                          json=data
                          )

        self.assertEqual(200, x.status_code, "Empty json provided")

        self.assertFalse(x.json()['cas_content'], "should be empty if no input is empty")

    def test_extract_definitions(self):
        json_filepath = os.path.join(ROOT, 'test_files/json/doc_bf4ef384-bd7a-51c8-8f7d-d2f61865d767.json')

        data = _get_json(json_filepath)

        x = requests.post(URL_extract_defs, json=data)

        self.assertCountEqual(['cas_content', 'content_type'], list(x.json().keys()),
                              'expected return keywords in json')

        self.assertTrue(x.json()['cas_content'], "check if not empty")
        self.assertTrue(x.json()['content_type'], "check if not empty")

        self.assertEqual(data['content_type'], x.json()['content_type'])

        s_in = data['cas_content']
        s_out = x.json()['cas_content']

        decoded_cas_content_in = _decode(s_in)
        decoded_cas_content_out = _decode(s_out)

        self.assertTrue(decoded_cas_content_in)
        self.assertTrue(decoded_cas_content_out)

    def test_small_nested_tables(self):
        json_filepath_in = os.path.join(ROOT, 'test_files/json/small_nested_tables.json')
        json_filepath_out = os.path.join(ROOT, 'test_files/response_json/small_nested_tables_response.json')

        data_in = _get_json(json_filepath_in)
        data_out = _get_json(json_filepath_out)

        x = requests.post(URL_extract_defs, json=data_in)

        self.assertEqual(data_out, x.json())

    def test_decode(self):
        json_filepath = os.path.join(ROOT, 'test_files/json/small_nested_tables.json')

        data = _get_json(json_filepath)

        decoded_cas_content_in = _decode(data['cas_content'])

        x = requests.post(URL_extract_defs, json=data)

        decoded_cas_content_out = _decode(x.json()['cas_content'])

        self.assertTrue(decoded_cas_content_in)
        self.assertTrue(decoded_cas_content_out)


def _decode(data_json):
    return base64.b64decode(data_json).decode('utf-8')


def _get_json(json_filepath):
    with open(json_filepath) as json_file:
        return json.load(json_file)


if __name__ == '__main__':
    unittest.main()
