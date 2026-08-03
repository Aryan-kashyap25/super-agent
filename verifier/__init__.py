"""Verification layer for LocalMind AI."""

from .confidence import clamp_confidence, compute_confidence
from .evidence_checker import EvidenceCheckResult, check_evidence
from .schema_validator import SourceReference, VerificationSchema, validate_verification_payload
from .verifier import CONFIDENCE_THRESHOLD, SAFE_FAILURE_MESSAGE, SafeFailureResponse, VerificationEngine, verify_generated_response

__all__ = [
	"CONFIDENCE_THRESHOLD",
	"EvidenceCheckResult",
	"SAFE_FAILURE_MESSAGE",
	"SafeFailureResponse",
	"SourceReference",
	"VerificationEngine",
	"VerificationSchema",
	"check_evidence",
	"clamp_confidence",
	"compute_confidence",
	"validate_verification_payload",
	"verify_generated_response",
]
