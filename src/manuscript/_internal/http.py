"""Streaming HTTP access for native Python and Pyodide, without global patches."""
import io
import sys
import urllib.error
import urllib.request


class _BrowserResponse(io.RawIOBase):
    def __init__(self, response, controller, run_sync, timeout):
        super().__init__()
        self.url, self.status = str(response.url), int(response.status)
        # Fetch may transparently decompress a body whose Content-Length is compressed.
        self.headers = {'content-length':None}
        self._controller, self._run_sync, self._timeout = controller, run_sync, timeout
        self._reader = response.body.getReader() if response.body is not None else None
        self._pending = bytearray()
        self._done = self._reader is None

    def readable(self):
        return True

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if self.closed:
            raise ValueError('Read from a closed HTTP response')
        if size == 0:
            return b''
        import asyncio
        async def read_chunk():
            try:
                return await self._reader.read()
            except Exception as exc:
                raise urllib.error.URLError(f'Browser response read failed: {exc}') from exc
        try:
            while not self._done and (size < 0 or len(self._pending) < size):
                chunk = self._run_sync(asyncio.wait_for(read_chunk(), self._timeout))
                if chunk.done:
                    self._done = True
                else:
                    self._pending.extend(chunk.value.to_py())
            count = len(self._pending) if size < 0 else min(size, len(self._pending))
            result = bytes(self._pending[:count])
            del self._pending[:count]
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            if not self._done:
                self._controller.abort()
            if self._reader is not None:
                try:
                    self._reader.releaseLock()
                except Exception:
                    pass  # An aborted read may still be settling in JavaScript.
            self._pending.clear()
            super().close()


def urlopen(url, timeout=None):
    """Open a stream; browser timeouts abort the underlying Fetch request."""
    if sys.platform != 'emscripten':
        return urllib.request.urlopen(url, timeout=timeout)
    from pyodide.ffi import can_run_sync, run_sync, to_js
    if not can_run_sync():
        raise RuntimeError('Browser downloads require JSPI and runPythonAsync()/callPromising().')
    import asyncio
    from js import AbortController, Object, fetch
    controller = AbortController.new()
    async def open_response():
        try:
            response = await fetch(str(url), to_js({'signal':controller.signal}, dict_converter=Object.fromEntries))
            if not response.ok:
                raise urllib.error.HTTPError(str(url), response.status, response.statusText, {}, None)
            return _BrowserResponse(response, controller, run_sync, timeout)
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            raise urllib.error.URLError(f'Browser fetch failed (check network/CORS): {exc}') from exc
    try:
        return run_sync(asyncio.wait_for(open_response(), timeout))
    except BaseException:
        controller.abort()
        raise


def sleep(seconds):
    """Back off without blocking the browser's JavaScript event loop."""
    if sys.platform != 'emscripten':
        import time
        time.sleep(seconds)
    else:
        import asyncio
        from pyodide.ffi import run_sync
        run_sync(asyncio.sleep(seconds))
