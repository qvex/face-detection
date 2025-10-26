from dataclasses import dataclass
from typing import TypeVar, Generic, Optional
from enum import Enum, auto
import numpy as np

from src.core.types import Result, Success, Failure
from src.core.errors import SessionError, SessionErrorKind
from src.verification.face_verifier import VerificationResult

State = TypeVar('State')

class SessionState(Enum):
    INITIAL = auto()
    CARD_PROCESSED = auto()
    LIVE_PROCESSED = auto()
    VERIFIED = auto()
    REJECTED = auto()

@dataclass(frozen=True, slots=True)
class SessionData:
    state: SessionState
    reference_embedding: Optional[np.ndarray]
    test_embedding: Optional[np.ndarray]
    verification_result: Optional[VerificationResult]

class VerificationSession(Generic[State]):
    def __init__(self, data: SessionData):
        self._data = data

    @property
    def state(self) -> SessionState:
        return self._data.state

    @property
    def reference_embedding(self) -> Optional[np.ndarray]:
        return self._data.reference_embedding

    @property
    def test_embedding(self) -> Optional[np.ndarray]:
        return self._data.test_embedding

    @property
    def verification_result(self) -> Optional[VerificationResult]:
        return self._data.verification_result

    def process_admit_card(
        self: 'VerificationSession[State]',
        embedding: np.ndarray
    ) -> Result['VerificationSession[State]', SessionError]:
        if self._data.state != SessionState.INITIAL:
            return Failure(SessionError(
                kind=SessionErrorKind.INVALID_STATE_TRANSITION,
                details=f"cannot process card from state {self._data.state.name}"
            ))

        new_data = SessionData(
            state=SessionState.CARD_PROCESSED,
            reference_embedding=embedding,
            test_embedding=None,
            verification_result=None
        )

        return Success(VerificationSession(new_data))

    def process_live_face(
        self: 'VerificationSession[State]',
        embedding: np.ndarray
    ) -> Result['VerificationSession[State]', SessionError]:
        if self._data.state != SessionState.CARD_PROCESSED:
            return Failure(SessionError(
                kind=SessionErrorKind.INVALID_STATE_TRANSITION,
                details=f"cannot process live face from state {self._data.state.name}"
            ))

        if self._data.reference_embedding is None:
            return Failure(SessionError(
                kind=SessionErrorKind.MISSING_REFERENCE_EMBEDDING,
                details="reference embedding not set"
            ))

        new_data = SessionData(
            state=SessionState.LIVE_PROCESSED,
            reference_embedding=self._data.reference_embedding,
            test_embedding=embedding,
            verification_result=None
        )

        return Success(VerificationSession(new_data))

    def complete_verification(
        self: 'VerificationSession[State]',
        result: VerificationResult
    ) -> Result['VerificationSession[State]', SessionError]:
        if self._data.state != SessionState.LIVE_PROCESSED:
            return Failure(SessionError(
                kind=SessionErrorKind.INVALID_STATE_TRANSITION,
                details=f"cannot complete verification from state {self._data.state.name}"
            ))

        final_state = SessionState.VERIFIED if result.is_match else SessionState.REJECTED

        new_data = SessionData(
            state=final_state,
            reference_embedding=self._data.reference_embedding,
            test_embedding=self._data.test_embedding,
            verification_result=result
        )

        return Success(VerificationSession(new_data))

def create_session() -> VerificationSession[SessionState]:
    initial_data = SessionData(
        state=SessionState.INITIAL,
        reference_embedding=None,
        test_embedding=None,
        verification_result=None
    )
    return VerificationSession(initial_data)
