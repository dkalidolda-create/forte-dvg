import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    api_key: str
    storage_root: Path
    analyzer_path: str = ''
    timeout_sec: int = 270
    max_upload_bytes: int = 100 * 1024 * 1024
    max_pending: int = 8

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env', override=False)
        storage = Path(os.getenv('STORAGE_ROOT', 'data/runtime'))
        if not storage.is_absolute():
            storage = ROOT / storage
        return cls(api_key=os.getenv('API_KEY', ''), storage_root=storage.resolve(),
                   analyzer_path=os.getenv('ANALYZER_PATH', ''),
                   timeout_sec=int(os.getenv('JOB_TIMEOUT_SEC', '270')),
                   max_upload_bytes=int(os.getenv('MAX_UPLOAD_MB', '100')) * 1024 * 1024,
                   max_pending=int(os.getenv('MAX_PENDING_JOBS', '8')))

    def validate(self):
        if len(self.api_key) < 24 or self.api_key == 'replace-with-generated-key':
            raise RuntimeError('Set API_KEY in .env to a random key of at least 24 characters')
        if not 1 <= self.timeout_sec <= 270:
            raise RuntimeError('JOB_TIMEOUT_SEC must be between 1 and 270')
        if self.max_upload_bytes <= 0 or self.max_pending <= 0:
            raise RuntimeError('Upload limit and queue capacity must be positive')
