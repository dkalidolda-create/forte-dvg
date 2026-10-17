import base64
import importlib
import math
from pathlib import Path

import cv2

from backend.app.domain.schemas import AnalysisResult, Candidate, Violation


def analyze_video(video_path: Path, analyzer_path: str):
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, _ = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not ok or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError('invalid_video')
        duration = count / fps
        if not analyzer_path:
            raise RuntimeError('model_not_configured')
        module, name = analyzer_path.split(':', 1)
        candidates = getattr(importlib.import_module(module), name)(video_path)
        if not isinstance(candidates, list):
            raise ValueError('invalid_model_output')
        events = []
        for raw in candidates:
            candidate = Candidate.model_validate(raw)
            if candidate.timecode_sec >= duration:
                raise ValueError('event_outside_video')
            evidence = []
            for timestamp in candidate.evidence_timestamps:
                if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
                    raise ValueError('evidence_outside_video')
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('evidence_decode_failed')
                actual_time = max(0, capture.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps
                height, width = frame.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError('evidence_encode_failed')
                evidence.append({'timestamp_sec': actual_time,
                                 'image_base64': base64.b64encode(jpeg.tobytes()).decode('ascii')})
            details = candidate.model_dump(exclude={'evidence_timestamps'})
            events.append(Violation(**details, evidence_frames=evidence))
        return AnalysisResult(verdict='violation' if events else 'no_violation',
                              violations=events).model_dump(exclude_none=True)
    finally:
        capture.release()
