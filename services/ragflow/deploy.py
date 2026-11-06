#!/usr/bin/env python3
"""Install an isolated AME service or upgrade an existing HTTP adapter."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[2]
INSTALL_ROOT = Path('/opt/ame-ragflow')
TOKEN_ROOT = Path('/etc/ame-ragflow')
UNIT_ROOT = Path('/etc/systemd/system')
LOCK_ROOT = Path('/run/lock')


class DeploymentError(Exception):
    pass


def run(*command, capture=False, env=None):
    if not capture:
        print('+ ' + shlex.join(str(item) for item in command), flush=True)
    result = subprocess.run([str(item) for item in command], check=True,
                            text=True, stdout=subprocess.PIPE if capture else None, env=env)
    return result.stdout.strip() if capture else ''


def property_value(service, name):
    return run('systemctl', 'show', service, '--property=' + name, '--value', capture=True)


def valid_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise DeploymentError('Invalid dataset ID')
    return value


def read_token(file):
    token = Path(file).read_text(encoding='utf-8').strip()
    if len(token) < 32 or any(c.isspace() or ord(c) < 32 for c in token):
        raise DeploymentError('Token file must contain a raw token of at least 32 characters')
    return token


def validate_corpus(file):
    chunks = json.loads(Path(file).read_text(encoding='utf-8'))
    if not isinstance(chunks, list) or not chunks:
        raise DeploymentError('Corpus must be a nonempty JSON array')
    ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or any(
                not isinstance(chunk.get(key), str) or not chunk[key].strip()
                for key in ('id', 'document', 'content')):
            raise DeploymentError('Each chunk needs nonempty id, document and content strings')
        if chunk['id'] in ids:
            raise DeploymentError('Duplicate corpus chunk ID')
        ids.add(chunk['id'])
    return len(chunks)


def atomic_copy(source, destination, mode=None):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix='.ame-update-', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


def request(port, route, body=None, token=None, timeout=3):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        headers = {'Content-Type': 'application/json'}
        if token is not None:
            headers['Authorization'] = 'Bearer ' + token
        connection.request('GET' if body is None else 'POST', route,
                           None if body is None else json.dumps(body), headers)
        response = connection.getresponse()
        body = json.loads(response.read(4 * 1024 * 1024))
        if not isinstance(body, dict):
            raise DeploymentError('Unexpected JSON response from the service')
        return response.status, body
    finally:
        connection.close()


def wait_ready(port, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            status, body = request(port, '/health', timeout=max(0.1, min(3, deadline - time.monotonic())))
            if status == 200 and body.get('ready') is True:
                return
        except (OSError, ValueError, http.client.HTTPException):
            pass
        time.sleep(1)
    raise DeploymentError('Health check timed out on port ' + str(port))


def verify_retrieval(port, dataset, token_file):
    status, body = request(port, '/api/v1/retrieval', {
        'dataset_ids': [dataset], 'question': 'test', 'page': 1, 'page_size': 3,
        'similarity_threshold': 0.2, 'knn_top_k': 64, 'knn_num_candidates': 128,
        'rerank_id': 'bge-reranker-v2-m3@maas-hw@OpenAI-API-Compatible',
        'rerank_candidates_count': 32,
    }, read_token(token_file), timeout=12)
    data = body.get('data')
    if (status != 200 or body.get('code') != 0 or not isinstance(data, dict)
            or not isinstance(data.get('chunks'), list)):
        raise DeploymentError('RAGFlow-compatible retrieval failed: HTTP ' + str(status))
    print('Retrieval verified: HTTP 200, chunks=' + str(len(data['chunks'])))


def print_result(service, port, dataset, token_file):
    print('Deployment succeeded')
    print('Service: ' + service)
    print('Port: ' + str(port))
    print('Dataset: ' + dataset)
    print('Token file: ' + str(token_file))
    print('Logs: sudo journalctl -u ' + service + ' -f')
    print('Configure the caller with the ECS private IP and this port.')


def ensure_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(('0.0.0.0', port))
        except OSError as error:
            raise DeploymentError('Port is occupied; use upgrade for an existing instance') from error


def render_unit(service, release, token_file, port, dataset):
    python = Path(sys.executable).resolve()
    if not re.fullmatch(r'/[A-Za-z0-9_./+@-]+', str(python)) or str(python).startswith(('/root/', '/home/')):
        raise DeploymentError('Use a system-wide Python installation outside home directories')
    return f'''[Unit]
Description=AME retrieval on port {port}
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
DynamicUser=yes
StateDirectory={service.removesuffix('.service')}
WorkingDirectory={release}
LoadCredential=token:{token_file}
Environment=AME_TOKEN_FILE=/run/credentials/{service}/token
Environment=AME_WORKER={release}/ame_worker
Environment=AME_CORPUS={release}/corpus.json
Environment=AME_DICTIONARY={release}/dictionary.txt
Environment=AME_CACHE=/var/lib/{service.removesuffix('.service')}/dictionary.cache
Environment=AME_DATASET_ID={dataset}
Environment=AME_BIND=0.0.0.0
Environment=AME_PORT={port}
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart={python} {release}/server.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
MemoryMax=2G
CPUQuota=100%
TasksMax=64
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
'''


def prepare_token(port, supplied):
    directory = TOKEN_ROOT / str(port)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    file = directory / 'token'
    if file.exists():
        existing = read_token(file)
        if supplied and read_token(supplied) != existing:
            raise DeploymentError('Existing token differs; refusing to rotate credentials')
    else:
        token = read_token(supplied) if supplied else secrets.token_urlsafe(48)
        fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            output.write(token + '\n')
    file.chmod(0o600)
    return file


def install(args):
    dataset = valid_id(args.dataset_id or ('ame-travel' if args.port == 18087 else 'ame-travel-' + str(args.port)))
    service = 'ame-ragflow-' + str(args.port) + '.service'
    unit = UNIT_ROOT / service
    if unit.exists() or property_value(service, 'LoadState') != 'not-found':
        raise DeploymentError('Service already exists; use upgrade with its service and file paths')
    ensure_port_free(args.port)
    print('Validated corpus: ' + str(validate_corpus(args.corpus)) + ' chunks')
    if args.token_file:
        read_token(args.token_file)
    for command in ('cmake', 'c++', 'make'):
        if shutil.which(command) is None:
            raise DeploymentError('Missing build dependency: ' + command)
    with tempfile.TemporaryDirectory(prefix='ame-build-') as temporary:
        build = Path(temporary)
        run('cmake', '-S', SOURCE, '-B', build, '-DAME_BUILD_RAGFLOW_SERVICE=ON',
            '-DCMAKE_BUILD_TYPE=Release', '-DPython3_EXECUTABLE=' + sys.executable)
        run('cmake', '--build', build, '--target', 'ame_worker', '--parallel', args.jobs)
        worker = build / 'services/ragflow/ame_worker'
        run(sys.executable, '-B', '-m', 'unittest', 'discover',
            '-s', SOURCE / 'services/ragflow/tests', '-p', 'test_server.py', '-v',
            env={**os.environ, 'AME_TEST_WORKER': str(worker)})
        run(sys.executable, '-B', '-m', 'unittest', 'discover',
            '-s', SOURCE / 'services/ragflow/tests', '-p', 'test_native.py', '-v',
            env={**os.environ, 'AME_TEST_WORKER': str(worker)})
        release = INSTALL_ROOT / str(args.port) / 'releases' / timestamp()
        release.mkdir(parents=True)
        for directory in (INSTALL_ROOT, release.parent.parent, release.parent, release):
            directory.chmod(0o755)
        for source, name, mode in (
            (worker, 'ame_worker', 0o755),
            (SOURCE / 'services/ragflow/server.py', 'server.py', 0o644),
            (SOURCE / 'lexicon/lexicon_src/segmentation/merged.txt', 'dictionary.txt', 0o644),
            (Path(args.corpus), 'corpus.json', 0o644),
        ):
            atomic_copy(source, release / name, mode)
    token_file = prepare_token(args.port, args.token_file)
    revision = run('git', '-C', SOURCE, 'rev-parse', 'HEAD', capture=True) if (SOURCE / '.git').exists() else None
    manifest = {
        'source_commit': revision, 'dataset_id': dataset, 'port': args.port,
        'sha256': {name: hashlib.sha256((release / name).read_bytes()).hexdigest()
                   for name in ('server.py', 'ame_worker', 'corpus.json', 'dictionary.txt')},
    }
    (release / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (release / 'manifest.json').chmod(0o644)
    unit.write_text(render_unit(service, release, token_file, args.port, dataset), encoding='utf-8')
    unit.chmod(0o644)
    try:
        run('systemctl', 'daemon-reload')
        run('systemctl', 'enable', '--now', service)
        wait_ready(args.port, args.wait)
        verify_retrieval(args.port, dataset, token_file)
    except BaseException:
        print('Installation failed; removing the new service. Release and token are retained.')
        try:
            run('systemctl', 'disable', '--now', service)
        finally:
            unit.unlink(missing_ok=True)
            run('systemctl', 'daemon-reload')
        raise
    print('Release: ' + str(release))
    print_result(service, args.port, dataset, token_file)


def upgrade(args):
    default_service = 'travel-ame' if args.port == 18087 else 'travel-ame-' + str(args.port)
    service = args.service or default_service
    if not service.endswith('.service'):
        service += '.service'
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.service', service):
        raise DeploymentError('Invalid service name')
    base = Path('/opt/travel-ame/current' if args.port == 18087 else '/opt/' + default_service)
    server = Path(args.server or base / 'server.py').resolve(strict=True)
    token_file = Path(args.token_file or '/etc/' + default_service + '/token')
    read_token(token_file)
    command = property_value(service, 'ExecStart')
    match = re.search(r'argv\[\]=(.*?)\s*;', command)
    argv = shlex.split(match.group(1)) if match else []
    if len(argv) != 2 or Path(argv[1]).resolve() != server:
        raise DeploymentError('ExecStart does not match the server path; no files changed')
    environment = dict(item.split('=', 1) for item in shlex.split(property_value(service, 'Environment')))
    if environment.get('AME_PORT') != str(args.port):
        raise DeploymentError('Service port does not match --port')
    if environment.get('AME_BIND', '127.0.0.1') not in ('127.0.0.1', '0.0.0.0'):
        raise DeploymentError('Automatic checks require AME_BIND=0.0.0.0 or 127.0.0.1')
    dataset = valid_id(environment.get('AME_DATASET_ID', 'ame-travel'))
    if property_value(service, 'ActiveState') != 'active':
        raise DeploymentError('Existing service must be active before upgrading')
    wait_ready(args.port, min(args.wait, 15))
    updated = SOURCE / 'services/ragflow/server.py'
    ast.parse(updated.read_text(encoding='utf-8'))
    backup = server.with_name(server.name + '.backup-' + timestamp())
    shutil.copy2(server, backup)
    print('Backup: ' + str(backup))
    try:
        atomic_copy(updated, server, server.stat().st_mode & 0o777)
        run('systemctl', 'restart', service)
        wait_ready(args.port, args.wait)
        verify_retrieval(args.port, dataset, token_file)
    except BaseException:
        print('Upgrade failed; restoring the previous adapter')
        atomic_copy(backup, server)
        run('systemctl', 'restart', service)
        wait_ready(args.port, args.wait)
        print('Previous adapter restored; inspect the service journal before retrying.')
        raise
    print_result(service, args.port, dataset, token_file)


@contextmanager
def deployment_lock(port):
    with (LOCK_ROOT / ('ame-ragflow-' + str(port) + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DeploymentError('Another deployment is running on this port') from error
        yield


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for name in ('install', 'upgrade'):
        command = commands.add_parser(name)
        command.add_argument('--port', type=int, default=18088)
        command.add_argument('--token-file', help='Existing raw token file; never a literal token')
        command.add_argument('--wait', type=int, default=180, help='Startup wait in seconds')
        if name == 'install':
            command.add_argument('--corpus', required=True)
            command.add_argument('--dataset-id')
            command.add_argument('--jobs', type=int, default=2)
        else:
            command.add_argument('--service', help='Override the existing service name')
            command.add_argument('--server', help='Override its existing server.py path')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.wait <= 600:
        parser.error('Port must be 1..65535 and wait must be 1..600')
    if args.operation == 'install' and not 1 <= args.jobs <= 32:
        parser.error('Jobs must be 1..32')
    return args


def main():
    args = arguments()
    if sys.version_info < (3, 10):
        raise DeploymentError('Python 3.10 or newer is required')
    if sys.platform != 'linux' or os.geteuid() != 0:
        raise DeploymentError('Run on the target Linux ECS with sudo')
    if shutil.which('systemctl') is None or not Path('/run/systemd/system').is_dir():
        raise DeploymentError('A running systemd installation is required')
    with deployment_lock(args.port):
        (install if args.operation == 'install' else upgrade)(args)


if __name__ == '__main__':
    try:
        main()
    except (DeploymentError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('Deployment interrupted', file=sys.stderr)
        sys.exit(130)
