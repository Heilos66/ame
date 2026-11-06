import argparse
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deploy


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'source'
        (self.source / 'services/ragflow').mkdir(parents=True)
        (self.source / 'services/ragflow/server.py').write_text('print("new")\n')
        dictionary = self.source / 'lexicon/lexicon_src/segmentation/merged.txt'
        dictionary.parent.mkdir(parents=True)
        dictionary.write_text('test\t0\tfixture\n')
        self.server = self.root / 'server.py'
        self.server.write_text('print("old")\n')
        self.token = self.root / 'token'
        self.token.write_text('deployment-test-token-' * 3 + '\n')
        self.corpus = self.root / 'corpus.json'
        self.corpus.write_text(json.dumps([{'id': '1', 'document': 'fixture', 'content': 'test'}]))
        self.units = self.root / 'units'
        self.units.mkdir()
        for name, value in {
            'SOURCE': self.source, 'INSTALL_ROOT': self.root / 'releases',
            'TOKEN_ROOT': self.root / 'tokens', 'UNIT_ROOT': self.units,
        }.items():
            manager = patch.object(deploy, name, value)
            manager.start()
            self.addCleanup(manager.stop)
        self.output = io.StringIO()
        manager = redirect_stdout(self.output)
        manager.__enter__()
        self.addCleanup(manager.__exit__, None, None, None)
        self.args = argparse.Namespace(port=18088, service=None, server=str(self.server),
                                       token_file=str(self.token), wait=3, jobs=1,
                                       corpus=str(self.corpus), dataset_id='ame-travel-18088')

    def properties(self, service, name):
        return {
            'LoadState': 'not-found', 'ActiveState': 'active',
            'ExecStart': '{ path=/usr/bin/python3 ; argv[]=/usr/bin/python3 ' + str(self.server) + ' ; }',
            'Environment': 'AME_PORT=18088 AME_DATASET_ID=ame-travel-18088 AME_BIND=0.0.0.0',
        }[name]

    @staticmethod
    def build(*command, **kwargs):
        if command[:2] == ('cmake', '--build'):
            binary = Path(command[2]) / 'services/ragflow/ame_worker'
            binary.parent.mkdir(parents=True)
            binary.write_text('test worker')
        return ''

    def test_install_keeps_token_private_and_release_independent_of_checkout(self):
        with patch.object(deploy, 'property_value', side_effect=self.properties), \
                patch.object(deploy, 'ensure_port_free'), \
                patch.object(deploy.shutil, 'which', return_value='/usr/bin/tool'), \
                patch.object(deploy, 'run', side_effect=self.build), \
                patch.object(deploy, 'wait_ready'), \
                patch.object(deploy, 'verify_retrieval') as verify:
            deploy.install(self.args)
        token_file = self.root / 'tokens/18088/token'
        self.assertEqual(0o600, token_file.stat().st_mode & 0o777)
        self.assertEqual(self.token.read_text(), token_file.read_text())
        unit = (self.units / 'ame-ragflow-18088.service').read_text()
        self.assertNotIn(str(self.source), unit)
        self.assertNotIn(self.token.read_text().strip(), self.output.getvalue())
        releases = list((self.root / 'releases/18088/releases').iterdir())
        self.assertEqual(1, len(releases))
        self.assertEqual(self.corpus.read_text(), (releases[0] / 'corpus.json').read_text())
        manifest = json.loads((releases[0] / 'manifest.json').read_text())
        self.assertIsNone(manifest['source_commit'])
        self.assertEqual(hashlib.sha256(self.corpus.read_bytes()).hexdigest(),
                         manifest['sha256']['corpus.json'])
        verify.assert_called_once_with(18088, 'ame-travel-18088', token_file)

    def test_install_failure_removes_only_the_new_unit(self):
        existing = self.units / 'travel-ame.service'
        existing.write_text('unrelated service')
        with patch.object(deploy, 'property_value', side_effect=self.properties), \
                patch.object(deploy, 'ensure_port_free'), \
                patch.object(deploy.shutil, 'which', return_value='/usr/bin/tool'), \
                patch.object(deploy, 'run', side_effect=self.build) as run, \
                patch.object(deploy, 'wait_ready'), \
                patch.object(deploy, 'verify_retrieval', side_effect=deploy.DeploymentError('HTTP 400')):
            with self.assertRaises(deploy.DeploymentError):
                deploy.install(self.args)
        self.assertFalse((self.units / 'ame-ragflow-18088.service').exists())
        self.assertEqual('unrelated service', existing.read_text())
        run.assert_any_call('systemctl', 'disable', '--now', 'ame-ragflow-18088.service')

    def test_install_refuses_to_replace_existing_service(self):
        unit = self.units / 'ame-ragflow-18088.service'
        unit.write_text('existing')
        with patch.object(deploy, 'run') as run:
            with self.assertRaises(deploy.DeploymentError):
                deploy.install(self.args)
        run.assert_not_called()
        self.assertEqual('existing', unit.read_text())

    def test_occupied_port_stops_installation_before_any_changes(self):
        with patch.object(deploy, 'property_value', side_effect=self.properties), \
                patch.object(deploy.socket, 'socket') as socket, patch.object(deploy, 'run') as run:
            socket.return_value.__enter__.return_value.bind.side_effect = OSError('Address in use')
            with self.assertRaisesRegex(deploy.DeploymentError, 'Port is occupied'):
                deploy.install(self.args)
        run.assert_not_called()
        self.assertEqual([], list(self.units.iterdir()))
        self.assertFalse((self.root / 'tokens').exists())

    def test_upgrade_backs_up_script_and_preserves_credentials(self):
        token = self.token.read_bytes()
        with patch.object(deploy, 'property_value', side_effect=self.properties), \
                patch.object(deploy, 'run') as run, patch.object(deploy, 'wait_ready'), \
                patch.object(deploy, 'verify_retrieval') as verify:
            deploy.upgrade(self.args)
        self.assertEqual('print("new")\n', self.server.read_text())
        backups = list(self.root.glob('server.py.backup-*'))
        self.assertEqual(1, len(backups))
        self.assertEqual('print("old")\n', backups[0].read_text())
        self.assertEqual(token, self.token.read_bytes())
        run.assert_called_once_with('systemctl', 'restart', 'travel-ame-18088.service')
        verify.assert_called_once_with(18088, 'ame-travel-18088', self.token)

    def test_failed_retrieval_rolls_upgrade_back(self):
        with patch.object(deploy, 'property_value', side_effect=self.properties), \
                patch.object(deploy, 'run') as run, patch.object(deploy, 'wait_ready'), \
                patch.object(deploy, 'verify_retrieval', side_effect=deploy.DeploymentError('HTTP 401')):
            with self.assertRaises(deploy.DeploymentError):
                deploy.upgrade(self.args)
        self.assertEqual('print("old")\n', self.server.read_text())
        self.assertEqual(2, run.call_count)
        self.assertNotIn(self.token.read_text().strip(), self.output.getvalue())

    def test_upgrade_refuses_an_unrelated_execstart(self):
        with patch.object(deploy, 'property_value', return_value='argv[]=/usr/bin/python3 /other/server.py ;'), \
                patch.object(deploy, 'run') as run:
            with self.assertRaises(deploy.DeploymentError):
                deploy.upgrade(self.args)
        self.assertEqual('print("old")\n', self.server.read_text())
        run.assert_not_called()
        self.assertEqual([], list(self.root.glob('server.py.backup-*')))

    def test_existing_token_is_not_rotated_on_retry(self):
        first = deploy.prepare_token(18088, None)
        original = first.read_bytes()
        self.assertEqual(first, deploy.prepare_token(18088, None))
        self.assertEqual(original, first.read_bytes())
        with self.assertRaises(deploy.DeploymentError):
            deploy.prepare_token(18088, self.token)
        self.assertEqual(original, first.read_bytes())

    def test_corpus_duplicates_are_rejected(self):
        chunk = {'id': 'same', 'document': 'fixture', 'content': 'test'}
        self.corpus.write_text(json.dumps([chunk, chunk]))
        with self.assertRaises(deploy.DeploymentError):
            deploy.validate_corpus(self.corpus)

    def test_verification_uses_ragflow_options_and_does_not_print_token(self):
        with patch.object(deploy, 'request', return_value=(200, {'code': 0, 'data': {'chunks': []}})) as request:
            deploy.verify_retrieval(18088, 'ame-travel-18088', self.token)
        body = request.call_args.args[2]
        self.assertEqual(0.2, body['similarity_threshold'])
        self.assertEqual(64, body['knn_top_k'])
        self.assertIn('rerank_id', body)
        self.assertNotIn(self.token.read_text().strip(), self.output.getvalue())

    def test_verification_rejects_malformed_success_response(self):
        with patch.object(deploy, 'request', return_value=(200, {'code': 0, 'data': None})):
            with self.assertRaisesRegex(deploy.DeploymentError, 'retrieval failed'):
                deploy.verify_retrieval(18088, 'ame-travel-18088', self.token)


if __name__ == '__main__':
    unittest.main()
