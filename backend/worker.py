"""Run separately with python -m backend.worker."""
import json
import logging
import multiprocessing
import tempfile
import time
from pathlib import Path

from backend.app.adapters.database.jobs import JobStore
from backend.app.config import Settings

log = logging.getLogger(__name__)


def execute(video_path, analyzer_path, output):
    try:
        from backend.app.pipeline.analyze import analyze_video
        payload = {'result': analyze_video(Path(video_path), analyzer_path)}
    except Exception as exc:
        log.exception('Video analysis failed')
        known = {'invalid_video', 'model_not_configured', 'invalid_model_output',
                 'event_outside_video', 'evidence_outside_video',
                 'evidence_decode_failed', 'evidence_encode_failed', 'analysis_uncertain',
                 'model_assets_missing', 'model_runtime_unavailable', 'analysis_timeout'}
        payload = {'error': str(exc) if str(exc) in known else 'analysis_failed'}
    Path(output).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')


def process_one(settings, store):
    job = store.claim()
    if job is None:
        return False
    remaining = job['created'] + settings.timeout_sec - time.time()
    if remaining <= 0:
        store.fail(job['id'], 'analysis_timeout')
        return True
    with tempfile.TemporaryDirectory(prefix='forte-result-') as temp:
        output = Path(temp) / 'result.json'
        child = multiprocessing.get_context('spawn').Process(
            target=execute, args=(job['path'], settings.analyzer_path, str(output)))
        try:
            child.start()
            child.join(timeout=remaining)
            if child.is_alive():
                store.fail(job['id'], 'analysis_timeout')
            elif child.exitcode != 0 or not output.exists():
                store.fail(job['id'], 'worker_process_failed')
            else:
                payload = json.loads(output.read_text(encoding='utf-8'))
                if 'error' in payload:
                    store.fail(job['id'], payload['error'])
                else:
                    store.complete(job['id'], payload['result'])
        except Exception:
            log.exception('Job failed: %s', job['id'])
            store.fail(job['id'], 'worker_process_failed')
        finally:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=3)
                    if child.is_alive():
                        child.kill()
                        child.join()
                child.close()
    return True


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings.load()
    settings.validate()
    store = JobStore(settings.storage_root / 'jobs.sqlite3', settings.timeout_sec)
    store.initialize()
    log.info('Worker started; analyzer: %s', settings.analyzer_path or 'NOT CONFIGURED')
    try:
        while True:
            if not process_one(settings, store):
                time.sleep(0.5)
    except KeyboardInterrupt:
        log.info('Worker stopped')


if __name__ == '__main__':
    main()
