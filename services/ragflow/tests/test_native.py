"""Exercise the HTTP adapter with the C++ worker built from this repository."""
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import Server, Worker
from test_server import RAGFLOW_OPTIONS, TOKEN


@unittest.skipUnless(os.environ.get('AME_TEST_WORKER'), 'Build with CMake and run CTest')
class NativeServiceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        corpus = root / 'corpus.json'
        dictionary = root / 'dictionary.txt'
        corpus.write_text(json.dumps([{
            'id': 'fixture-lamp', 'document': 'fixture',
            'content': '长信宫灯是汉代铜灯，博物馆展品。',
        }], ensure_ascii=False), encoding='utf-8')
        dictionary.write_text('长信宫灯\t0\ttest\n汉代\t0\ttest\n'
                              '铜灯\t0\ttest\n博物馆\t0\ttest\n', encoding='utf-8')
        worker = Worker(os.environ['AME_TEST_WORKER'], corpus, dictionary,
                        root / 'dictionary.cache', startup_timeout=10)
        self.addCleanup(worker.close)
        self.server = Server(('127.0.0.1', 0), worker, TOKEN, 'ame-travel')
        self.addCleanup(self.server.server_close)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.shutdown)

    def request(self, extra):
        client = http.client.HTTPConnection(*self.server.server_address, timeout=10)
        try:
            client.request('POST', '/api/v1/retrieval', json.dumps({
                'dataset_ids': ['ame-travel'], 'question': '长信宫灯',
                'page': 1, 'page_size': 3, **extra,
            }), {'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'})
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def test_ragflow_body_recalls_the_same_corpus(self):
        status, baseline = self.request({})
        self.assertEqual(200, status)
        self.assertEqual(['fixture-lamp'], [c['id'] for c in baseline['data']['chunks']])
        status, result = self.request(RAGFLOW_OPTIONS)
        self.assertEqual(200, status)
        self.assertEqual(0, result['code'])
        self.assertEqual('ame-graph', result['meta']['backend'])
        self.assertEqual(['fixture-lamp'], [c['id'] for c in result['data']['chunks']])
        self.assertIn('汉代铜灯', result['data']['chunks'][0]['content'])


if __name__ == '__main__':
    unittest.main()
