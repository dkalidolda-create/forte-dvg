from pathlib import Path
from typing import List, Protocol

from backend.app.domain.schemas import Candidate


class Analyzer(Protocol):
    def __call__(self, video_path: Path) -> List[Candidate]:
        """Run locally. Return [] only after actual successful analysis."""
        ...
