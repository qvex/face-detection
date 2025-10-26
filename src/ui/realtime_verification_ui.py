from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from src.verification.reference_processor import ReferenceImage
from src.verification.face_verifier import VerificationResult

@dataclass(frozen=True, slots=True)
class RealtimeUIConfig:
    window_name: str = "Real-Time Verification"
    window_width: int = 1280
    window_height: int = 720
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_large: float = 1.2
    font_scale_medium: float = 0.8
    font_scale_small: float = 0.6
    font_thickness: int = 2
    color_match: tuple = (0, 255, 0)
    color_no_match: tuple = (0, 0, 255)
    color_white: tuple = (255, 255, 255)
    color_blue: tuple = (255, 165, 0)
    match_threshold: float = 0.4

class RealtimeVerificationUI:
    def __init__(self, config: RealtimeUIConfig = RealtimeUIConfig()):
        self._config = config
        self._camera: Optional[cv2.VideoCapture] = None
        self._reference_image: Optional[np.ndarray] = None

    def _initialize_camera(self) -> bool:
        self._camera = cv2.VideoCapture(0)
        return self._camera.isOpened()

    def _release_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()

    def _draw_split_screen(
        self,
        reference_display: np.ndarray,
        live_frame: np.ndarray
    ) -> np.ndarray:
        canvas = np.zeros(
            (self._config.window_height, self._config.window_width, 3),
            dtype=np.uint8
        )

        ref_height, ref_width = reference_display.shape[:2]
        live_height, live_width = live_frame.shape[:2]

        half_width = self._config.window_width // 2

        ref_resized = cv2.resize(reference_display, (half_width, self._config.window_height))
        live_resized = cv2.resize(live_frame, (half_width, self._config.window_height))

        canvas[:, :half_width] = ref_resized
        canvas[:, half_width:] = live_resized

        cv2.line(
            canvas,
            (half_width, 0),
            (half_width, self._config.window_height),
            self._config.color_white,
            2
        )

        return canvas

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        color: tuple,
        scale: float
    ) -> None:
        cv2.putText(
            frame,
            text,
            position,
            self._config.font,
            scale,
            color,
            self._config.font_thickness
        )

    def _draw_reference_label(self, frame: np.ndarray, is_admit_card: bool) -> None:
        label = "ADMIT CARD" if is_admit_card else "REFERENCE PHOTO"
        self._draw_text(frame, label, (20, 40), self._config.color_white, self._config.font_scale_small)

    def _draw_live_label(self, frame: np.ndarray) -> None:
        x_offset = self._config.window_width // 2 + 20
        self._draw_text(frame, "LIVE CAMERA", (x_offset, 40), self._config.color_white, self._config.font_scale_small)

    def _draw_verification_result(
        self,
        frame: np.ndarray,
        result: Optional[VerificationResult]
    ) -> None:
        y_center = self._config.window_height // 2

        if result is None:
            status_text = "NO FACE DETECTED"
            status_color = self._config.color_blue
            similarity_text = "---"
        elif result.is_match:
            status_text = "MATCH"
            status_color = self._config.color_match
            similarity_text = f"{result.similarity_score:.3f}"
        else:
            status_text = "NO MATCH"
            status_color = self._config.color_no_match
            similarity_text = f"{result.similarity_score:.3f}"

        text_size = cv2.getTextSize(
            status_text,
            self._config.font,
            self._config.font_scale_large,
            self._config.font_thickness
        )[0]

        x = (self._config.window_width - text_size[0]) // 2
        y = y_center - 50

        self._draw_text(
            frame,
            status_text,
            (x, y),
            status_color,
            self._config.font_scale_large
        )

        similarity_label = f"Similarity: {similarity_text}"
        sim_size = cv2.getTextSize(
            similarity_label,
            self._config.font,
            self._config.font_scale_medium,
            self._config.font_thickness
        )[0]

        x_sim = (self._config.window_width - sim_size[0]) // 2
        y_sim = y_center + 20

        self._draw_text(
            frame,
            similarity_label,
            (x_sim, y_sim),
            self._config.color_white,
            self._config.font_scale_medium
        )

    def _draw_instructions(self, frame: np.ndarray) -> None:
        self._draw_text(
            frame,
            "Press Q to quit",
            (20, self._config.window_height - 20),
            self._config.color_white,
            self._config.font_scale_small
        )

    def initialize(self, reference: ReferenceImage) -> bool:
        success = self._initialize_camera()
        if not success:
            return False

        self._reference_image = reference.original_image.copy()

        bbox = reference.face_detection.bbox
        cv2.rectangle(
            self._reference_image,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            self._config.color_match,
            2
        )

        cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self._config.window_name,
            self._config.window_width,
            self._config.window_height
        )

        return True

    def show_frame(
        self,
        live_frame: np.ndarray,
        verification_result: Optional[VerificationResult],
        is_admit_card: bool
    ) -> bool:
        display = self._draw_split_screen(self._reference_image, live_frame)

        self._draw_reference_label(display, is_admit_card)
        self._draw_live_label(display)
        self._draw_verification_result(display, verification_result)
        self._draw_instructions(display)

        cv2.imshow(self._config.window_name, display)

        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._camera.isOpened():
            return None

        ret, frame = self._camera.read()
        return frame if ret else None

    def cleanup(self) -> None:
        self._release_camera()
        cv2.destroyAllWindows()

def create_realtime_ui(
    config: Optional[RealtimeUIConfig] = None
) -> RealtimeVerificationUI:
    actual_config = config if config else RealtimeUIConfig()
    return RealtimeVerificationUI(actual_config)
