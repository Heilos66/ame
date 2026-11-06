#!/usr/bin/env python3
"""Bounded, authenticated RAGFlow-compatible HTTP adapter for one AME graph worker."""
import hmac
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 262144  # Also accepts 10000 characters escaped as surrogate pairs.
MAX_QUERY_CHARS = 10000
MAX_WORKER_RESPONSE_BYTES = 4 * 1024 * 1024


class WorkerFailure(Exception):
    pass


class Busy(Exception):
    pass


class Worker:
    def __init__(self, binary, corpus, dictionary, cache, startup_timeout=120):
        self.process = subprocess.Popen(
            [str(binary), str(corpus), str(dictionary), str(cache)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        for pipe in (self.process.stdin, self.process.stdout):
            os.set_blocking(pipe.fileno(), False)
        self.lock = threading.Lock()
        try:
            ready = self._read(time.monotonic() + startup_timeout)
            if ready.get('ready') is not True:
                raise WorkerFailure()
            self.chunks = ready['chunks']
        except Exception:
            self.close()
            raise

    def healthy(self):
        return self.process.poll() is None

    def _wait(self, pipe, event, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerFailure()
        with selectors.DefaultSelector() as selector:
            selector.register(pipe, event)
            if not selector.select(remaining):
                raise WorkerFailure()

    def _read(self, deadline):
        data = bytearray()
        while len(data) <= MAX_WORKER_RESPONSE_BYTES:
            self._wait(self.process.stdout, selectors.EVENT_READ, deadline)
            part = os.read(self.process.stdout.fileno(), 65536)
            if not part:
                raise WorkerFailure()
            data.extend(part)
            if b'\n' in part:
                result = json.loads(data)
                if not isinstance(result, dict) or 'error' in result:
                    raise WorkerFailure()
                return result
        raise WorkerFailure()

    def query(self, question, limit, deadline):
        queued = time.monotonic()
        if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise Busy()
        queue_ms = (time.monotonic() - queued) * 1000
        try:
            data = memoryview(json.dumps({'query': question, 'limit': limit}).encode() + b'\n')
            while data:
                self._wait(self.process.stdin, selectors.EVENT_WRITE, deadline)
                size = os.write(self.process.stdin.fileno(), data)
                data = data[size:]
            result = self._read(deadline)
            result["queue_ms"] = queue_ms
            return result
        except (OSError, ValueError, WorkerFailure) as error:
            # A timed-out reply must never become the next request's result.
            self.close()
            raise WorkerFailure() from error
        finally:
            self.lock.release()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()


def validate(body, dataset):
    if not isinstance(body, dict) or body.get('dataset_ids') != [dataset]:
        raise ValueError()
    question, limit = body.get('question'), body.get('page_size', 3)
    if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUERY_CHARS:
        raise ValueError()
    # Reject lone surrogate escapes before passing JSON to native code.
    question.encode('utf-8')
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError()
    if body.get('page', 1) != 1:
        raise ValueError()
    # Accept but ignore RAGFlow tuning fields: similarity_threshold, rerank_id,
    # knn_top_k, knn_num_candidates and rerank_candidates_count. AME uses its
    # graph recall settings and scores, not RAGFlow's vector/rerank parameters.
    # Only the validated question and limit are forwarded to the worker.
    return question, limit


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, worker, token, dataset, request_timeout=8, max_connections=32):
        super().__init__(address, Handler)
        self.worker, self.token, self.dataset = worker, token.encode(), dataset
        self.request_timeout = request_timeout
        self.connections = threading.BoundedSemaphore(max_connections)
        self.queue = threading.BoundedSemaphore(8)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def handle_error(self, request, client_address):
        # No request path, headers, bodies, peer addresses or exception text in logs.
        print('ame_http status=connection_failed', flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = 'AME'
    sys_version = ''

    def setup(self):
        super().setup()
        self.connection.settimeout(5)
        self.deadline = time.monotonic() + self.server.request_timeout

    def log_message(self, *_args):
        pass

    def reply(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/health':
            self.reply(404, {'code': 404})
        else:
            healthy = self.server.worker.healthy()
            self.reply(200 if healthy else 503, {'ready': healthy})

    def do_POST(self):
        started = time.monotonic()
        if self.path != '/api/v1/retrieval':
            self.reply(404, {'code': 404})
            return
        provided = self.headers.get('Authorization', '').encode('utf-8')
        if not hmac.compare_digest(provided, b'Bearer ' + self.server.token):
            self.reply(401, {'code': 401, 'message': 'unauthorized'})
            return
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError()
            size = int(lengths[0])
            if not 1 <= size <= MAX_BODY_BYTES:
                self.reply(413, {'code': 413})
                return
            raw = bytearray()
            while len(raw) < size:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                self.connection.settimeout(min(5, remaining))
                part = self.rfile.read1(min(65536, size - len(raw)))
                if not part:
                    raise ValueError()
                raw.extend(part)
            question, limit = validate(json.loads(raw), self.server.dataset)
        except (ValueError, UnicodeError, RecursionError):
            self.reply(400, {'code': 400, 'message': 'invalid request'})
            return
        except (TimeoutError, OSError):
            self.reply(408, {'code': 408})
            return
        if not self.server.queue.acquire(blocking=False):
            self.reply(503, {'code': 503, 'message': 'busy'})
            return
        try:
            result = self.server.worker.query(question, limit, self.deadline)
            chunks = [{'id': hit['id'], 'content': hit['content'],
                       'document_keyword': hit['document'], 'similarity': hit['score']}
                      for hit in result['hits']]
            elapsed = (time.monotonic() - started) * 1000
            recall = result['recall_ms']
            if not math.isfinite(recall) or any(not math.isfinite(c['similarity']) for c in chunks):
                raise WorkerFailure()
            meta = {'backend': 'ame-graph', 'ame_recall_ms': recall,
                    'adapter_ms': elapsed,
                    'queue_ms': result['queue_ms']}
            self.reply(200, {'code': 0, 'data': {'chunks': chunks}, 'meta': meta})
            print(f'ame_retrieval status=succeeded chunks={len(chunks)} elapsed_ms={elapsed:.3f}',
                  flush=True)
        except Busy:
            self.reply(503, {'code': 503, 'message': 'busy'})
        except (WorkerFailure, KeyError, TypeError, ValueError):
            self.reply(502, {'code': 502, 'message': 'retrieval unavailable'})
        finally:
            self.server.queue.release()


def main():
    token = Path(os.environ['AME_TOKEN_FILE']).read_text().strip()
    dataset = os.environ.get('AME_DATASET_ID', 'ame-travel')
    if len(token) < 32 or any(c.isspace() or ord(c) < 32 for c in token):
        raise ValueError('invalid AME token file')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', dataset):
        raise ValueError('invalid AME dataset ID')
    worker = Worker(os.environ['AME_WORKER'], os.environ['AME_CORPUS'],
                    os.environ['AME_DICTIONARY'], os.environ['AME_CACHE'])
    server = Server((os.environ.get('AME_BIND', '127.0.0.1'),
                     int(os.environ.get('AME_PORT', '18087'))), worker, token, dataset)
    def stop(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    def monitor():
        while worker.healthy():
            time.sleep(1)
        stop()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threading.Thread(target=monitor, daemon=True).start()
    print(f'ame_service status=ready chunks={worker.chunks}', flush=True)
    try:
        server.serve_forever()
    finally:
        healthy = worker.healthy()
        server.server_close()
        worker.close()
    if not healthy:
        raise SystemExit(1)  # systemd restarts and rebuilds the read-only graph.


if __name__ == '__main__':
    main()
