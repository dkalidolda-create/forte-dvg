import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.request_guard import APIRequestGuard
from tests.test_backend import multipart


class APIRequestGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def call(self, chunks=(), headers=(), key=b'k' * 32, max_upload_bytes=8,
             path='/api/v1/analyze', method='POST'):
        received = 0

        async def run():
            nonlocal received
            app = create_app(Settings('k' * 32, self.root, max_upload_bytes=max_upload_bytes))
            request_headers = list(headers)
            if key is not None:
                request_headers.append((b'x-api-key', key))
            scope = {
                'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                'query_string': b'', 'root_path': '', 'headers': request_headers,
                'client': ('127.0.0.1', 1), 'server': ('test', 80),
            }
            messages = []

            async def receive():
                nonlocal received
                if received >= len(chunks):
                    raise AssertionError('Request body must not be read again')
                body = chunks[received]
                received += 1
                return {'type': 'http.request', 'body': body, 'more_body': received < len(chunks)}

            async def send(message):
                messages.append(message)

            async with app.router.lifespan_context(app):
                await app(scope, receive, send)
            code = next(m['status'] for m in messages if m['type'] == 'http.response.start')
            body = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
            return code, json.loads(body)

        code, result = asyncio.run(run())
        return code, result, received

    def test_invalid_key_is_rejected_without_reading_body(self):
        for key in (None, b'wrong'):
            code, body, received = self.call(key=key)
            self.assertEqual(code, 401)
            self.assertEqual(body, {'detail': 'Invalid or missing X-API-Key'})
            self.assertEqual(received, 0)

    def test_job_lookup_requires_key_before_any_body(self):
        code, _, received = self.call(key=None, method='GET', path='/api/v1/jobs/not-a-uuid')
        self.assertEqual(code, 401)
        self.assertEqual(received, 0)

    def test_declared_oversized_body_is_rejected_before_reading(self):
        limit = 8 + APIRequestGuard.MULTIPART_OVERHEAD_BYTES
        code, body, received = self.call(headers=[(b'content-length', str(limit + 1).encode())])
        self.assertEqual(code, 413)
        self.assertEqual(body, {'detail': 'Video exceeds upload limit'})
        self.assertEqual(received, 0)

    def test_streamed_multipart_stops_at_total_body_limit(self):
        prefix, content_type = multipart('test.mov', b'')
        prefix = prefix.split(b'\r\n--forte-test-boundary--')[0]
        too_large = b'x' * (APIRequestGuard.MULTIPART_OVERHEAD_BYTES + 8)
        code, body, received = self.call(
            chunks=[prefix, too_large, b'not consumed'],
            headers=[(b'content-type', content_type.encode())])
        self.assertEqual(code, 413)
        self.assertEqual(body, {'detail': 'Video exceeds upload limit'})
        self.assertEqual(received, 2)
        self.assertEqual(list((self.root / 'uploads').glob('*')), [])

    def test_29_mib_mov_is_accepted_with_normal_file_limit(self):
        data = b'x' * (29 * 1024 * 1024)
        body, content_type = multipart('test.mov', data)
        code, result, received = self.call(
            chunks=[body], max_upload_bytes=100 * 1024 * 1024,
            headers=[(b'content-type', content_type.encode()),
                     (b'content-length', str(len(body)).encode())])
        self.assertEqual(code, 202)
        self.assertEqual(result['status'], 'processing')
        self.assertEqual(received, 1)
        uploaded = list((self.root / 'uploads').glob('*.mov'))
        self.assertEqual(len(uploaded), 1)
        self.assertEqual(uploaded[0].stat().st_size, len(data))


if __name__ == '__main__':
    unittest.main()
