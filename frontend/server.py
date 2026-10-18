"""Local web UI and same-origin API gateway. The API key never reaches the browser."""
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from backend.app.domain.schemas import AcceptedJob, AnalysisResult, JobResponse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'


@dataclass(frozen=True)
class UISettings:
    api_key: str
    backend_url: str = 'http://127.0.0.1:8000'
    max_upload_bytes: int = 100 * 1024 * 1024

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env', override=False)
        return cls(os.getenv('API_KEY', ''), os.getenv('BACKEND_URL', 'http://127.0.0.1:8000'),
                   int(os.getenv('MAX_UPLOAD_MB', '100')) * 1024 * 1024)


class RequestGuard:
    """Limit raw multipart bytes before spooling; reject cross-site browser writes."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        if scope['method'] == 'POST':
            origin = headers.get(b'origin', b'').decode()
            expected = scope['scheme'] + '://' + headers.get(b'host', b'').decode()
            if (origin and origin != expected) or headers.get(b'sec-fetch-site') == b'cross-site':
                return await JSONResponse({'detail': 'Запрос с другого сайта запрещён.'}, 403)(scope, receive, send)
            try:
                size = int(headers.get(b'content-length', b'0'))
            except ValueError:
                size = self.limit + 1
            if size > self.limit:
                return await JSONResponse({'detail': 'Файл превышает допустимый размер.'}, 413)(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > self.limit:
                    raise HTTPException(413, 'Файл превышает допустимый размер.')
            return message

        await self.app(scope, limited_receive, send)


ERRORS = {
    400: 'Файл пуст или запрос не удалось прочитать.',
    401: 'Сервис ещё не настроен для анализа. Обратитесь к команде проекта.',
    404: 'Задание не найдено. Загрузите видео ещё раз.',
    413: 'Файл превышает допустимый размер.',
    415: 'Поддерживаются видео MOV и MP4.',
    422: 'Не удалось прочитать файл или номер задания.',
    429: 'Сейчас обрабатывается слишком много видео. Попробуйте немного позже.',
}


def create_app(settings=None, client_factory=None):
    settings = settings or UISettings.load()
    if not settings.backend_url.startswith(('http://', 'https://')):
        raise ValueError('BACKEND_URL must be an HTTP(S) URL')
    if settings.max_upload_bytes <= 0:
        raise ValueError('Upload limit must be positive')
    if client_factory is None:
        def client_factory():
            return httpx.Client(base_url=settings.backend_url, timeout=httpx.Timeout(30, connect=3),
                                trust_env=False, follow_redirects=False)
    app = FastAPI(title='Forte DVG · Web', docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestGuard, limit=settings.max_upload_bytes + 1024 * 1024)

    @app.middleware('http')
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
            "font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        if request.url.path.startswith('/ui-api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/ui-api/config')
    def config():
        return {'max_upload_bytes': settings.max_upload_bytes, 'formats': ['mov', 'mp4'],
                'poll_interval_ms': 2500, 'wait_timeout_ms': 300000}

    def call_backend(method, path, **kwargs):
        if len(settings.api_key) < 24:
            raise HTTPException(503, ERRORS[401])
        try:
            with client_factory() as client:
                response = client.request(method, path, headers={'X-API-Key': settings.api_key}, **kwargs)
        except httpx.TimeoutException:
            raise HTTPException(504, 'Сервис анализа долго не отвечает. Попробуйте снова.') from None
        except httpx.RequestError:
            raise HTTPException(503, 'Сервис анализа недоступен. Попробуйте позже.') from None
        if response.status_code not in {200, 202}:
            raise HTTPException(response.status_code if response.status_code in ERRORS else 502,
                                ERRORS.get(response.status_code, 'Сервис анализа вернул ошибку.'))
        try:
            return response.json()
        except ValueError:
            raise HTTPException(502, 'Сервис вернул некорректный ответ. Анализ не подтверждён.') from None

    @app.post('/ui-api/analyze', status_code=202)
    def analyze(file: UploadFile):
        try:
            suffix = Path(file.filename or '').suffix.lower()
            if suffix not in {'.mov', '.mp4'}:
                raise HTTPException(415, ERRORS[415])
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            if not size:
                raise HTTPException(400, ERRORS[400])
            if size > settings.max_upload_bytes:
                raise HTTPException(413, ERRORS[413])
            payload = call_backend('POST', '/api/v1/analyze',
                                   files={'file': (Path(file.filename).name, file.file,
                                                   'video/quicktime' if suffix == '.mov' else 'video/mp4')})
            try:
                result = AcceptedJob.model_validate(payload)
                UUID(result.job_id)
                return result.model_dump()
            except (ValidationError, ValueError, TypeError):
                raise HTTPException(502, 'Сервис не вернул корректный номер задания.') from None
        finally:
            file.file.close()

    @app.get('/ui-api/jobs/{job_id}')
    def job(job_id: UUID):
        payload = call_backend('GET', '/api/v1/jobs/' + str(job_id))
        try:
            result = JobResponse.model_validate(payload)
            if result.job_id != str(job_id):
                raise ValueError('Wrong job ID')
            if result.status == 'done':
                analysis = AnalysisResult.model_validate(payload)
                for event in analysis.violations:
                    if (not math.isfinite(event.timecode_sec) or event.timecode_sec < 0
                            or any(not math.isfinite(f.timestamp_sec) or f.timestamp_sec < 0
                                   for f in event.evidence_frames)):
                        raise ValueError('Invalid event timestamp')
            return result.model_dump(exclude_none=True)
        except (ValidationError, ValueError, TypeError):
            raise HTTPException(502, 'Результат анализа неполный. Вывод не может быть показан.') from None

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app
