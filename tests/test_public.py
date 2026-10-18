import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.public import create_public_app
from frontend.server import UISettings
from frontend.server import create_app as create_web_app


class PublicGatewayTests(unittest.TestCase):
    def test_browser_upload_does_not_depend_on_development_api_port(self):
        key = 'synthetic-long-key-for-public-test'
        job = '30af1522-2f17-4b45-8d2c-66f6de77a812'

        def reachable_public_api(request):
            if request.url.port == 8000:
                raise httpx.ConnectError('Development server is stopped', request=request)
            self.assertEqual(request.url.port, 9000)
            self.assertEqual(request.headers['X-API-Key'], key)
            return httpx.Response(202, json={'job_id': job, 'status': 'processing'})

        def factory():
            return httpx.Client(base_url='http://127.0.0.1:9000',
                                transport=httpx.MockTransport(reachable_public_api))

        # Capture the factory's actual destination instead of hiding it behind
        # the mock client's own base_url.
        from frontend.server import create_app as web

        def mounted(ui, _factory):
            return web(ui, lambda: httpx.Client(base_url=ui.backend_url,
                       transport=httpx.MockTransport(reachable_public_api)))

        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {
                'BACKEND_URL': 'http://127.0.0.1:8000', 'PUBLIC_PORT': '9000',
                'API_KEY': 'stale-ui-key-must-not-be-used'}):
            os.environ.pop('PUBLIC_INTERNAL_API_URL', None)
            with patch('backend.public.create_web_app', side_effect=mounted):
                app = create_public_app(Settings(key, Path(tmp)), factory)
            with TestClient(app) as client:
                response = client.post('/ui-api/analyze', files={'file': ('a.mov', b'test')})
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.json()['job_id'], job)

    def test_one_origin_serves_ui_and_protected_jury_api(self):
        key = 'synthetic-long-key-for-public-test'
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(Settings(key, Path(tmp)))
            app.mount('/', create_web_app(UISettings(key)), name='web')
            with TestClient(app) as client:
                self.assertEqual(client.get('/').status_code, 200)
                self.assertEqual(client.get('/ui-api/config').status_code, 200)
                self.assertEqual(client.get('/health').json(), {'status': 'ok'})
                self.assertEqual(client.get('/openapi.json').status_code, 200)
                response = client.post('/api/v1/analyze', files={'file': ('a.mov', b'test')})
                self.assertEqual(response.status_code, 401)
                response = client.post('/api/v1/analyze', headers={'X-API-Key': key},
                                       files={'file': ('a.mov', b'test')})
                self.assertEqual(response.status_code, 202)
