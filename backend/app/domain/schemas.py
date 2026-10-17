from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Candidate(BaseModel):
    """Model output; evidence is extracted by the pipeline from the source video."""
    type: str = Field(min_length=1)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_section: Optional[str] = None
    evidence_timestamps: List[float] = Field(min_length=1, max_length=3)


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: List[EvidenceFrame] = Field(min_length=1, max_length=3)
    rule: str
    source_title: str
    source_url: str
    source_section: Optional[str] = None


class AnalysisResult(BaseModel):
    verdict: Literal['violation', 'no_violation']
    violations: List[Violation]

    @model_validator(mode='after')
    def consistent_verdict(self):
        if (self.verdict == 'violation') != bool(self.violations):
            raise ValueError('Verdict must agree with events')
        return self


class AcceptedJob(BaseModel):
    job_id: str
    status: Literal['processing'] = 'processing'


class JobResponse(BaseModel):
    job_id: str
    status: Literal['processing', 'done', 'failed']
    verdict: Optional[Literal['violation', 'no_violation']] = None
    violations: Optional[List[Violation]] = None
    error: Optional[str] = None
