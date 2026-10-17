"""Durable queue shared by API and worker on one machine."""
import json
import sqlite3
import time
from contextlib import contextmanager


class JobStore:
    def __init__(self, path, timeout_sec):
        self.path, self.timeout = path, timeout_sec

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, path TEXT NOT NULL, state TEXT NOT NULL,
                created REAL NOT NULL, result TEXT, error TEXT)''')

    def expire(self):
        with self.connect() as db:
            db.execute('''UPDATE jobs SET state='failed', error='analysis_timeout'
                WHERE state IN ('queued','running') AND created < ?''',
                       (time.time() - self.timeout,))

    def enqueue(self, job_id, path, max_pending):
        self.expire()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
            if count >= max_pending:
                return False
            db.execute("INSERT INTO jobs(id,path,state,created) VALUES (?,?,'queued',?)",
                       (job_id, str(path), time.time()))
        return True

    def claim(self):
        self.expire()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (row['id'],))
            return dict(row)

    def complete(self, job_id, result):
        self.expire()
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='done', result=? WHERE id=? AND state='running'",
                       (json.dumps(result, ensure_ascii=False), job_id))

    def fail(self, job_id, error):
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='failed', error=? WHERE id=? AND state IN ('queued','running')",
                       (error, job_id))

    def get(self, job_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            return None
        result = {'job_id': row['id'], 'status': 'processing'}
        if row['state'] == 'done':
            result.update(status='done', **json.loads(row['result']))
        elif row['state'] == 'failed':
            result.update(status='failed', error=row['error'])
        return result
