import asyncio
import base64
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from backend.app.adapters.database.jobs import JobStore
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.pipeline.analyze import analyze_video
from backend.worker import process_one


async def request(app, method, path, body=b'', key='k' * 32, content_type=None):
    """Exercise the ASGI app without opening a network port."""
    headers = []
    if key is not None:
        headers.append((b'x-api-key', key.encode()))
    if content_type:
        headers.append((b'content-type', content_type.encode()))
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {'type': 'http.request', 'body': body, 'more_body': False}
        await asyncio.sleep(0)
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)

    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
             'method': method, 'scheme': 'http', 'path': path,
             'raw_path': path.encode(), 'query_string': b'', 'root_path': '',
             'headers': headers, 'client': ('127.0.0.1', 1), 'server': ('test', 80)}
    await app(scope, receive, send)
    status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
    response = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    return status, json.loads(response)


def multipart(filename, data):
    boundary = 'forte-test-boundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\nContent-Type: video/quicktime\r\n\r\n').encode()
    return body + data + f'\r\n--{boundary}--\r\n'.encode(), f'multipart/form-data; boundary={boundary}'


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings('k' * 32, self.root)
        self.store = JobStore(self.root / 'jobs.sqlite3', 270)
        self.store.initialize()
        self.video = self.root / 'synthetic.mov'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
        writer.release()

    def tearDown(self):
        self.temp.cleanup()

    def call(self, method, path, body=b'', key='k' * 32, content_type=None, settings=None):
        async def run():
            app = create_app(settings or self.settings)
            async with app.router.lifespan_context(app):
                return await request(app, method, path, body, key, content_type)
        return asyncio.run(run())

    def upload(self, filename='video.mov', data=None, **kwargs):
        body, ct = multipart(filename, self.video.read_bytes() if data is None else data)
        return self.call('POST', '/api/v1/analyze', body, content_type=ct, **kwargs)

    def test_authentication_and_health(self):
        self.assertEqual(self.call('GET', '/health', key=None)[0], 200)
        for key in [None, 'wrong']:
            self.assertEqual(self.upload(key=key)[0], 401)
            self.assertEqual(self.call('GET', '/api/v1/jobs/' + str(uuid4()), key=key)[0], 401)

    def test_upload_lookup_and_persistence(self):
        code, job = self.upload()
        self.assertEqual(code, 202)
        code, found = self.call('GET', '/api/v1/jobs/' + job['job_id'])
        self.assertEqual(code, 200)
        self.assertEqual(found['status'], 'processing')
        reopened = JobStore(self.root / 'jobs.sqlite3', 270)
        self.assertIsNotNone(reopened.get(job['job_id']))
        self.assertEqual(len(list((self.root / 'uploads').glob('*.mov'))), 1)

    def test_upload_validation_and_cleanup(self):
        self.assertEqual(self.upload(filename='x.txt')[0], 415)
        self.assertEqual(self.upload(data=b'')[0], 400)
        self.assertEqual(self.upload(settings=replace(self.settings, max_upload_bytes=2))[0], 413)
        self.assertEqual(list((self.root / 'uploads').glob('*')), [])

    def test_unknown_and_malformed_job(self):
        self.assertEqual(self.call('GET', '/api/v1/jobs/' + str(uuid4()))[0], 404)
        self.assertEqual(self.call('GET', '/api/v1/jobs/not-a-uuid')[0], 422)

    def test_queue_capacity_and_atomic_claim(self):
        settings = replace(self.settings, max_pending=1)
        self.assertEqual(self.upload(settings=settings)[0], 202)
        self.assertEqual(self.upload(settings=settings)[0], 429)
        self.assertIsNotNone(self.store.claim())
        self.assertIsNone(self.store.claim())
        self.assertEqual(len(list((self.root / 'uploads').glob('*'))), 1)

    def test_expired_job_cannot_be_overwritten(self):
        job_id = str(uuid4())
        self.store.enqueue(job_id, self.video, 8)
        self.store.claim()
        with self.store.connect() as db:
            db.execute('UPDATE jobs SET created=? WHERE id=?', (time.time() - 300, job_id))
        self.store.complete(job_id, {'verdict': 'no_violation', 'violations': []})
        self.assertEqual(self.store.get(job_id)['error'], 'analysis_timeout')

    def test_no_model_is_not_a_negative_verdict(self):
        with self.assertRaisesRegex(RuntimeError, 'model_not_configured'):
            analyze_video(self.video, '')

    def test_corrupt_video(self):
        path = self.root / 'corrupt.mov'
        path.write_bytes(b'not a video')
        with self.assertRaisesRegex(ValueError, 'invalid_video'):
            analyze_video(path, 'tests.analyzers:empty')

    def test_real_jpeg_and_rule_fields(self):
        result = analyze_video(self.video, 'tests.analyzers:event')
        self.assertEqual(result['verdict'], 'violation')
        frame = result['violations'][0]['evidence_frames'][0]
        jpeg = base64.b64decode(frame['image_base64'], validate=True)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR))
        self.assertAlmostEqual(frame['timestamp_sec'], 0.2)
        self.assertIn('source_title', result['violations'][0])

    def test_no_violation_and_invalid_time(self):
        self.assertEqual(analyze_video(self.video, 'tests.analyzers:empty')['verdict'], 'no_violation')
        with self.assertRaisesRegex(ValueError, 'evidence_outside_video'):
            analyze_video(self.video, 'tests.analyzers:outside')

    def test_worker_end_to_end(self):
        _, job = self.upload()
        process_one(replace(self.settings, analyzer_path='tests.analyzers:event'), self.store)
        _, result = self.call('GET', '/api/v1/jobs/' + job['job_id'])
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['verdict'], 'violation')

    def test_worker_timeout(self):
        job_id = str(uuid4())
        self.store.enqueue(job_id, self.video, 8)
        process_one(replace(self.settings, analyzer_path='tests.analyzers:slow', timeout_sec=2), self.store)
        self.assertEqual(self.store.get(job_id)['error'], 'analysis_timeout')

    def test_worker_missing_model_fails(self):
        job_id = str(uuid4())
        self.store.enqueue(job_id, self.video, 8)
        process_one(self.settings, self.store)
        result = self.store.get(job_id)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('verdict', result)
        self.assertEqual(result['error'], 'model_not_configured')


if __name__ == '__main__':
    unittest.main()
