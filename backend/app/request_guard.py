"""Reject unauthorized or oversized API requests before multipart parsing."""

import secrets

from fastapi import HTTPException
from starlette.responses import JSONResponse


class APIRequestGuard:
    # Multipart boundaries and file headers are separate from the video's limit.
    MULTIPART_OVERHEAD_BYTES = 1024 * 1024

    def __init__(self, app, api_key, max_upload_bytes):
        self.app = app
        self.api_key = api_key.encode()
        self.max_body_bytes = max_upload_bytes + self.MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        path = scope['path'].rstrip('/')
        upload = scope['method'] == 'POST' and path == '/api/v1/analyze'
        lookup = scope['method'] == 'GET' and path.startswith('/api/v1/jobs/')
        if not (upload or lookup):
            return await self.app(scope, receive, send)

        headers = scope.get('headers', [])
        keys = [value for name, value in headers if name.lower() == b'x-api-key']
        if (len(keys) != 1 or not keys[0]
                or not secrets.compare_digest(keys[0], self.api_key)):
            return await JSONResponse({'detail': 'Invalid or missing X-API-Key'}, 401)(
                scope, receive, send)
        if not upload:
            return await self.app(scope, receive, send)

        lengths = [value for name, value in headers if name.lower() == b'content-length']
        try:
            declared = int(lengths[0]) if lengths else 0
        except ValueError:
            declared = -1
        if len(lengths) > 1 or declared < 0 or declared > self.max_body_bytes:
            return await JSONResponse({'detail': 'Video exceeds upload limit'}, 413)(
                scope, receive, send)

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > self.max_body_bytes:
                    raise HTTPException(413, 'Video exceeds upload limit')
            return message

        return await self.app(scope, limited_receive, send)
