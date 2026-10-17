import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

from backend.app.adapters.database.jobs import JobStore
from backend.app.config import Settings
from backend.app.domain.schemas import AcceptedJob, JobResponse


def create_app(settings=None):
    settings = settings or Settings.load()
    store = JobStore(settings.storage_root / 'jobs.sqlite3', settings.timeout_sec)

    @asynccontextmanager
    async def lifespan(app):
        settings.validate()
        (settings.storage_root / 'uploads').mkdir(parents=True, exist_ok=True)
        store.initialize()
        yield

    app = FastAPI(title='Forte Video Analysis API', version='0.2.0', lifespan=lifespan)
    header = APIKeyHeader(name='X-API-Key', auto_error=False)

    def authorize(key: Annotated[Optional[str], Depends(header)]):
        if not key or not secrets.compare_digest(key.encode(), settings.api_key.encode()):
            raise HTTPException(401, 'Invalid or missing X-API-Key')

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.post('/api/v1/analyze', response_model=AcceptedJob, status_code=202,
              dependencies=[Depends(authorize)])
    def analyze(file: Annotated[UploadFile, File()]):
        path, accepted = None, False
        try:
            suffix = Path(file.filename or '').suffix.lower()
            if suffix not in {'.mov', '.mp4'}:
                raise HTTPException(415, 'Upload a .mov or .mp4 file')
            job_id = str(uuid4())
            path = settings.storage_root / 'uploads' / (job_id + suffix)
            size = 0
            with path.open('xb') as target:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, 'Video exceeds upload limit')
                    target.write(chunk)
            if not size:
                raise HTTPException(400, 'Empty video')
            if not store.enqueue(job_id, path, settings.max_pending):
                raise HTTPException(429, 'Queue is full; retry later')
            accepted = True
            return {'job_id': job_id, 'status': 'processing'}
        finally:
            file.file.close()
            if path is not None and not accepted:
                path.unlink(missing_ok=True)

    @app.get('/api/v1/jobs/{job_id}', response_model=JobResponse,
             response_model_exclude_none=True, dependencies=[Depends(authorize)])
    def get_job(job_id: UUID):
        store.expire()
        result = store.get(str(job_id))
        if result is None:
            raise HTTPException(404, 'Job not found')
        return result

    return app


app = create_app()
