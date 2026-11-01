"""Serve the browser demo and registry artifacts from the same origin."""
import argparse
import copy
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen


CHUNK_SIZE = 1024 * 1024


def artifact_id(artifact):
    return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()


def create_handler(directory, cache, catalog):
    directory, cache = Path(directory), Path(cache)
    artifacts = {artifact_id(spec): spec for entry in catalog['models'].values()
                 for spec in entry['artifacts'].values() if spec and spec.get('urls')}
    cache.mkdir(parents=True, exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def end_headers(self):
            path = urlsplit(self.path).path
            if path == '/' or path.endswith(('.html', '.mjs', '.py', '.json')):
                self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/catalog.json':
                host = self.headers.get('Host', '')
                if not re.fullmatch(r'[a-zA-Z0-9.\-:\[\]]+', host):
                    self.send_error(400)
                    return
                result = copy.deepcopy(catalog)
                for entry in result['models'].values():
                    for spec in entry['artifacts'].values():
                        if spec and spec.get('urls'):
                            spec['urls'] = [f'http://{host}/_artifacts/{artifact_id(spec)}']
                body = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            elif path.startswith('/_artifacts/'):
                identifier = path.removeprefix('/_artifacts/')
                spec = artifacts.get(identifier)
                if spec is None:
                    self.send_error(404, 'Unknown registry artifact')
                    return
                self.serve_artifact(identifier, spec)
            else:
                super().do_GET()

        def serve_artifact(self, identifier, spec):
            target = cache / identifier
            if target.is_file():
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(target.stat().st_size))
                self.end_headers()
                try:
                    with target.open('rb') as source:
                        shutil.copyfileobj(source, self.wfile, CHUNK_SIZE)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            source = None
            for url in spec['urls']:
                try:
                    source = urlopen(url, timeout=30)
                    break
                except OSError as exc:
                    self.log_error('Cannot fetch %s: %s', spec['filename'], exc)
            if source is None:
                self.send_error(502, 'Registry mirrors could not provide this artifact')
                return
            fd, temporary = tempfile.mkstemp(dir=cache, suffix='.part')
            try:
                with source, os.fdopen(fd, 'wb') as output:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    if spec.get('size') is not None:
                        self.send_header('Content-Length', str(spec['size']))
                    self.end_headers()
                    digest, size = hashlib.sha256(), 0
                    read_chunk = getattr(source, 'read1', source.read)
                    while chunk := read_chunk(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        self.wfile.write(chunk)
                    if spec.get('size') is not None and size != spec['size']:
                        raise ValueError('Artifact size mismatch')
                    if spec.get('sha256') and digest.hexdigest() != spec['sha256']:
                        raise ValueError('Artifact SHA-256 mismatch')
                os.replace(temporary, target)
            except (OSError, ValueError) as exc:
                self.log_error('Download interrupted for %s: %s', spec['filename'], exc)
                self.close_connection = True
            finally:
                Path(temporary).unlink(missing_ok=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/private/tmp/manuscript-pyodide-check'))
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/manuscript-browser-download-cache'))
    args = parser.parse_args()
    catalog = json.loads((args.directory / 'catalog.json').read_text())
    server = ThreadingHTTPServer((args.bind, args.port), create_handler(args.directory, args.cache, catalog))
    print(f'Browser demo: http://{args.bind}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
