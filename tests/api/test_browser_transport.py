"""Streaming and cancellation without a browser or network requests."""
import asyncio
from types import SimpleNamespace
import urllib.error

import pytest

from manuscript._internal.http import _BrowserResponse


class Reader:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.reads = 0
        self.released = False

    async def read(self):
        self.reads += 1
        chunk = next(self.chunks, None)
        if isinstance(chunk, Exception):
            raise chunk
        return SimpleNamespace(done=chunk is None, value=SimpleNamespace(to_py=lambda: chunk))

    def releaseLock(self):
        self.released = True


def response(chunks):
    reader = Reader(chunks)
    aborted = []
    controller = SimpleNamespace(abort=lambda: aborted.append(True))
    raw = SimpleNamespace(url='https://example.test/model', status=200,
                          body=SimpleNamespace(getReader=lambda: reader))
    return _BrowserResponse(raw, controller, asyncio.run, 1), reader, aborted


def test_stream_consumes_only_needed_chunks_and_preserves_pending_bytes():
    stream, reader, aborted = response([b'abc', b'def', b'gh'])
    with stream:
        assert stream.read(0) == b''
        assert reader.reads == 0
        assert stream.read(2) == b'ab'
        assert reader.reads == 1
        assert stream.read(2) == b'cd'
        assert reader.reads == 2
        assert stream.read() == b'efgh'
        assert stream.read(1) == b''
    assert reader.released
    assert not aborted


def test_close_aborts_incomplete_download_and_is_idempotent():
    stream, reader, aborted = response([b'abc', b'def'])
    assert stream.read(1) == b'a'
    stream.close()
    stream.close()
    assert aborted == [True]
    assert reader.released
    with pytest.raises(ValueError, match='closed HTTP response'):
        stream.read()


def test_read_failure_aborts_and_releases_reader():
    stream, reader, aborted = response([RuntimeError('connection lost')])
    with pytest.raises(urllib.error.URLError, match='connection lost'):
        stream.read(1)
    assert stream.closed
    assert aborted == [True]
    assert reader.released


def test_read_timeout_aborts_underlying_fetch():
    stream, reader, aborted = response([])
    async def stalled_read():
        await asyncio.sleep(10)
    reader.read = stalled_read
    stream._timeout = .001
    with pytest.raises(asyncio.TimeoutError):
        stream.read(1)
    assert stream.closed
    assert aborted == [True]
    assert reader.released
