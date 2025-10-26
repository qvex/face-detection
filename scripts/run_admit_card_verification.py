import sys
import numpy as np

from src.detection.insightface_detector import create_insightface_detector
from src.recognition.insightface_cpu_recognizer import create_insightface_cpu_recognizer
from src.verification.face_verifier import create_face_verifier
from src.verification.verification_session import create_session
from src.ui.verification_ui import create_verification_ui
from src.core.types import Success, Failure

def process_card_stage(detector, recognizer, ui, session):
    ui.show_processing("Initializing card capture...")

    card_frame = ui.show_card_capture()
    if card_frame is None:
        return None, None

    ui.show_processing("Detecting face on admit card...")

    detection_result = detector.detect(card_frame)
    if isinstance(detection_result, Failure):
        error_msg = f"Card detection failed: {detection_result.error.details}"
        return None, error_msg

    face_detection = detection_result.value

    ui.show_processing("Extracting face embedding from card...")

    embedding_result = recognizer.extract_embedding(face_detection.aligned_face)
    if isinstance(embedding_result, Failure):
        error_msg = f"Embedding extraction failed: {embedding_result.error.details}"
        return None, error_msg

    card_embedding = embedding_result.value

    session_result = session.process_admit_card(card_embedding)
    if isinstance(session_result, Failure):
        error_msg = f"Session error: {session_result.error.details}"
        return None, error_msg

    return session_result.value, None

def process_live_stage(detector, recognizer, ui, session):
    ui.show_processing("Initializing live capture...")

    live_frame = ui.show_live_capture()
    if live_frame is None:
        return None, None

    ui.show_processing("Detecting live face...")

    detection_result = detector.detect(live_frame)
    if isinstance(detection_result, Failure):
        error_msg = f"Live detection failed: {detection_result.error.details}"
        return None, error_msg

    face_detection = detection_result.value

    ui.show_processing("Extracting live face embedding...")

    embedding_result = recognizer.extract_embedding(face_detection.aligned_face)
    if isinstance(embedding_result, Failure):
        error_msg = f"Embedding extraction failed: {embedding_result.error.details}"
        return None, error_msg

    live_embedding = embedding_result.value

    session_result = session.process_live_face(live_embedding)
    if isinstance(session_result, Failure):
        error_msg = f"Session error: {session_result.error.details}"
        return None, error_msg

    return session_result.value, None

def process_verification_stage(verifier, ui, session):
    ui.show_processing("Computing similarity...")

    verification_result = verifier.verify(
        session.reference_embedding,
        session.test_embedding
    )

    if isinstance(verification_result, Failure):
        error_msg = f"Verification failed: {verification_result.error.details}"
        return None, error_msg

    result = verification_result.value

    session_result = session.complete_verification(result)
    if isinstance(session_result, Failure):
        error_msg = f"Session error: {session_result.error.details}"
        return None, error_msg

    return session_result.value, result

def run_verification_loop():
    detector = create_insightface_detector(model_name='buffalo_l', confidence_threshold=0.5)
    recognizer = create_insightface_cpu_recognizer(model_name='buffalo_l')
    verifier = create_face_verifier(threshold=0.4)
    ui = create_verification_ui()

    if not ui.initialize():
        print("Failed to initialize camera")
        return 1

    should_continue = True

    while should_continue:
        session = create_session()

        updated_session, error = process_card_stage(
            detector,
            recognizer,
            ui,
            session
        )

        if error:
            action = ui.show_error(error)
            if action == 'quit':
                should_continue = False
            continue

        if updated_session is None:
            should_continue = False
            continue

        session = updated_session

        updated_session, error = process_live_stage(
            detector,
            recognizer,
            ui,
            session
        )

        if error:
            action = ui.show_error(error)
            if action == 'quit':
                should_continue = False
            continue

        if updated_session is None:
            should_continue = False
            continue

        session = updated_session

        final_session, verification_result = process_verification_stage(
            verifier,
            ui,
            session
        )

        if verification_result is None:
            action = ui.show_error(error)
            if action == 'quit':
                should_continue = False
            continue

        action = ui.show_result(verification_result)

        if action == 'quit':
            should_continue = False

    ui.cleanup()
    return 0

if __name__ == "__main__":
    sys.exit(run_verification_loop())
