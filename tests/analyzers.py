"""Synthetic predictors for tests ONLY. Never configure these for jury videos."""
import time


def empty(path):
    return []


def event(path):
    return [{'type': 'TEST ONLY', 'timecode_sec': 0.2,
             'description': 'Synthetic fixture; not a real detector.',
             'rule': 'Test rule', 'source_title': 'Test source',
             'source_url': 'https://example.invalid', 'evidence_timestamps': [0.2]}]


def outside(path):
    result = event(path)
    result[0]['evidence_timestamps'] = [9999]
    return result


def slow(path):
    time.sleep(30)
    return []
