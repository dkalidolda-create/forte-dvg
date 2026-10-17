import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from ml.datasets.category_manifest import prepare


class CategoryManifestTests(unittest.TestCase):
    def test_withdrawn_category_is_excluded_without_changing_original(self):
        rows = [{'video_id': key, 'video_path': f'/{key}.mov', 'split': 'train',
                 'sha256': key, 'target': target, 'event_sec': '9' if target == '1' else ''}
                for key, target in [('X-15', '1'), ('X-26', '1'), ('X-5', '0')]]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'original.csv', Path(directory)/'category.csv'
            with source.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(source, output)
            with output.open(encoding='utf-8-sig') as file:
                result = list(csv.DictReader(file))
            self.assertEqual({row['video_id']: row['target'] for row in result},
                             {'X-26': '1', 'X-5': '0'})
            self.assertEqual(source.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
