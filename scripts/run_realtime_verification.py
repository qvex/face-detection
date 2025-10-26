import sys
from pathlib import Path

from src.detection.mtcnn_detector import create_mtcnn_detector
from src.recognition.insightface_cpu_recognizer import create_insightface_cpu_recognizer
from src.verification.face_verifier import create_face_verifier
from src.verification.reference_processor import create_reference_processor
from src.ui.realtime_verification_ui import create_realtime_ui
from src.ui.image_browser import create_image_browser
from src.core.types import Failure

def print_usage():
    print("Real-Time Face Verification")
    print()
    print("Usage: python scripts/run_realtime_verification.py [reference_image_path]")
    print()
    print("Arguments:")
    print("  reference_image_path    Optional path to reference image (face photo or admit card)")
    print("                          If omitted, interactive file browser will be shown")
    print()
    print("Examples:")
    print("  python scripts/run_realtime_verification.py")
    print("  python scripts/run_realtime_verification.py data/reference/student_photo.jpg")

def run_realtime_verification(image_path: str = None) -> int:
    if image_path is None:
        browser = create_image_browser()
        browser_result = browser.select_image()

        if isinstance(browser_result, Failure):
            print(f"Error: {browser_result.error.details}")
            return 1

        reference_path = browser_result.value
    else:
        reference_path = Path(image_path)

        if not reference_path.exists():
            print(f"Error: Image file not found: {reference_path}")
            return 1

    print("Initializing models...")
    detector = create_mtcnn_detector(min_face_size=40, confidence_threshold=0.9)
    recognizer = create_insightface_cpu_recognizer(model_name='buffalo_l')
    verifier = create_face_verifier(threshold=0.4)

    print(f"Loading reference image: {reference_path}")
    processor = create_reference_processor(detector, recognizer)

    reference_result = processor.process_reference(reference_path)

    if isinstance(reference_result, Failure):
        print(f"Error processing reference image: {reference_result.error.details}")
        return 1

    reference = reference_result.value

    image_type = "admit card" if reference.is_admit_card else "face photo"
    print(f"Reference image type detected: {image_type}")
    print()
    print("Starting real-time verification...")
    print("Press Q to quit")
    print()

    ui = create_realtime_ui()

    if not ui.initialize(reference):
        print("Error: Failed to initialize camera")
        return 1

    try:
        should_continue = True

        while should_continue:
            frame = ui.read_frame()

            if frame is None:
                print("Error: Failed to read camera frame")
                break

            detection_result = detector.detect(frame)

            if isinstance(detection_result, Failure):
                should_continue = ui.show_frame(
                    frame,
                    None,
                    reference.is_admit_card
                )
                continue

            face_detection = detection_result.value

            embedding_result = recognizer.extract_embedding(
                face_detection.aligned_face
            )

            if isinstance(embedding_result, Failure):
                should_continue = ui.show_frame(
                    frame,
                    None,
                    reference.is_admit_card
                )
                continue

            live_embedding = embedding_result.value

            verification_result = verifier.verify(
                reference.embedding,
                live_embedding
            )

            if isinstance(verification_result, Failure):
                should_continue = ui.show_frame(
                    frame,
                    None,
                    reference.is_admit_card
                )
                continue

            result = verification_result.value

            should_continue = ui.show_frame(
                frame,
                result,
                reference.is_admit_card
            )

    finally:
        ui.cleanup()

    print("Real-time verification stopped")
    return 0

def main():
    image_path = sys.argv[1] if len(sys.argv) >= 2 else None
    return run_realtime_verification(image_path)

if __name__ == "__main__":
    sys.exit(main())
