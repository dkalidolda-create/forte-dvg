import unittest

import httpx
from fastapi.testclient import TestClient

from frontend.server import UISettings, create_app

JOB = '30af1522-2f17-4b45-8d2c-66f6de77a812'
KEY = 'synthetic-api-key-for-tests-only'


class FrontendTests(unittest.TestCase):
    def client(self, handler, max_bytes=1024):
        def factory():
            return httpx.Client(base_url='http://backend.test', transport=httpx.MockTransport(handler))
        return TestClient(create_app(UISettings(KEY, max_upload_bytes=max_bytes), factory))

    def test_static_page_and_config_do_not_leak_server_key(self):
        with self.client(lambda request: httpx.Response(500)) as client:
            page = client.get('/')
            self.assertEqual(page.status_code, 200)
            self.assertNotIn(KEY, page.text)
            self.assertIn('Forte', page.text)
            self.assertIn("script-src 'self'", page.headers['content-security-policy'])
            self.assertEqual(client.get('/ui-api/config').json()['max_upload_bytes'], 1024)
            self.assertNotIn(KEY, client.get('/ui-api/config').text)
            self.assertEqual(client.get('/static/app.js').status_code, 200)

    def test_upload_forwards_key_server_side_and_preserves_job_id(self):
        def backend(request):
            self.assertEqual(request.headers['X-API-Key'], KEY)
            self.assertIn(b'clip.mov', request.read())
            return httpx.Response(202, json={'job_id': JOB, 'status': 'processing'})
        with self.client(backend) as client:
            response = client.post('/ui-api/analyze', files={'file': ('clip.mov', b'test-video')})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()['job_id'], JOB)
            self.assertNotIn(KEY, response.text)

    def test_file_validation_and_cross_site_rejection_precede_backend(self):
        def forbidden(request):
            self.fail('Invalid file reached backend')
        with self.client(forbidden) as client:
            for name, content, status in [('a.txt', b'abc', 415), ('a.mov', b'', 400), ('a.mp4', b'x'*1025, 413)]:
                self.assertEqual(client.post('/ui-api/analyze', files={'file': (name, content)}).status_code, status)
            response = client.post('/ui-api/analyze', headers={'Origin': 'https://other.test'}, files={'file': ('a.mov', b'x')})
            self.assertEqual(response.status_code, 403)
            response = client.post('/ui-api/analyze', headers={'Content-Length': '2000000'}, content=b'x')
            self.assertEqual(response.status_code, 413)

    def test_chunked_body_is_limited_before_multipart_spooling_completes(self):
        def forbidden(request):
            self.fail('Oversize body reached backend')
        def body():
            yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="a.mov"\r\n\r\n'
            for _ in range(20):
                yield b'x'*65536
            yield b'\r\n--test--\r\n'
        with self.client(forbidden) as client:
            response = client.post('/ui-api/analyze', headers={'Content-Type': 'multipart/form-data; boundary=test'}, content=body())
            self.assertEqual(response.status_code, 413)

    def test_completed_result_must_have_consistent_verdict_and_matching_id(self):
        cases = [({'job_id': JOB, 'status': 'done'}, 502),
                 ({'job_id': JOB, 'status': 'done', 'verdict': 'violation', 'violations': []}, 502),
                 ({'job_id': JOB, 'status': 'done', 'verdict': 'no_violation', 'violations': []}, 200),
                 ({'job_id': JOB, 'status': 'failed', 'error': 'model_not_configured'}, 200)]
        for payload, status in cases:
            with self.client(lambda request: httpx.Response(200, json=payload)) as client:
                response = client.get('/ui-api/jobs/'+JOB)
                self.assertEqual(response.status_code, status)
                if payload['status'] == 'failed':
                    self.assertNotIn('verdict', response.json())

    def test_backend_error_and_key_are_not_reflected_as_a_verdict(self):
        with self.client(lambda request: httpx.Response(401, json={'detail': KEY})) as client:
            response = client.get('/ui-api/jobs/'+JOB)
            self.assertEqual(response.status_code, 401)
            self.assertNotIn(KEY, response.text)
            self.assertNotIn('verdict', response.json())

    def test_network_failure_becomes_a_clear_service_error(self):
        def unavailable(request):
            raise httpx.ConnectError('connection failed', request=request)
        with self.client(unavailable) as client:
            response = client.get('/ui-api/jobs/'+JOB)
            self.assertEqual(response.status_code, 503)
            self.assertIn('недоступен', response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
