from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
import cv2
import numpy as np

from src.verification.verification_session import VerificationSession, SessionState
from src.verification.face_verifier import VerificationResult

class UIState(Enum):
    CARD_CAPTURE = auto()
    CARD_PROCESSING = auto()
    LIVE_CAPTURE = auto()
    LIVE_PROCESSING = auto()
    RESULT_DISPLAY = auto()
    ERROR_DISPLAY = auto()

@dataclass(frozen=True, slots=True)
class UIColors:
    white: tuple = (255, 255, 255)
    green: tuple = (0, 255, 0)
    red: tuple = (0, 0, 255)
    blue: tuple = (255, 165, 0)
    gray: tuple = (128, 128, 128)
    black: tuple = (0, 0, 0)

@dataclass(frozen=True, slots=True)
class UIConfig:
    window_name: str = "Admit Card Verification"
    window_width: int = 1280
    window_height: int = 720
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2
    guide_thickness: int = 3
    colors: UIColors = UIColors()

class VerificationUI:
    def __init__(self, config: UIConfig = UIConfig()):
        self._config = config
        self._ui_state = UIState.CARD_CAPTURE
        self._error_message: Optional[str] = None
        self._camera: Optional[cv2.VideoCapture] = None

    def _initialize_camera(self) -> bool:
        self._camera = cv2.VideoCapture(0)
        return self._camera.isOpened()

    def _release_camera(self) -> None:
        if self._camera is not None:
            self._camera.release()

    def _create_blank_frame(self) -> np.ndarray:
        return np.zeros(
            (self._config.window_height, self._config.window_width, 3),
            dtype=np.uint8
        )

    def _draw_text_centered(
        self,
        frame: np.ndarray,
        text: str,
        y_position: int,
        color: tuple
    ) -> None:
        text_size = cv2.getTextSize(
            text,
            self._config.font,
            self._config.font_scale,
            self._config.font_thickness
        )[0]

        x = (self._config.window_width - text_size[0]) // 2

        cv2.putText(
            frame,
            text,
            (x, y_position),
            self._config.font,
            self._config.font_scale,
            color,
            self._config.font_thickness
        )

    def _draw_card_guide_rectangle(self, frame: np.ndarray) -> None:
        width = 600
        height = 400
        x = (self._config.window_width - width) // 2
        y = (self._config.window_height - height) // 2

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            self._config.colors.blue,
            self._config.guide_thickness
        )

    def _draw_stage_indicator(
        self,
        frame: np.ndarray,
        stage: int,
        total_stages: int
    ) -> None:
        text = f"Stage {stage}/{total_stages}"
        self._draw_text_centered(frame, text, 50, self._config.colors.white)

    def _draw_instructions(
        self,
        frame: np.ndarray,
        instructions: list[str],
        start_y: int
    ) -> None:
        for i, instruction in enumerate(instructions):
            y_pos = start_y + (i * 40)
            self._draw_text_centered(
                frame,
                instruction,
                y_pos,
                self._config.colors.white
            )

    def _render_card_capture_screen(self, camera_frame: np.ndarray) -> np.ndarray:
        display_frame = camera_frame.copy()

        self._draw_stage_indicator(display_frame, 1, 3)
        self._draw_card_guide_rectangle(display_frame)

        instructions = [
            "Place admit card within the blue rectangle",
            "Press SPACE to capture",
            "Press Q to quit"
        ]
        self._draw_instructions(display_frame, instructions, 600)

        return display_frame

    def _render_live_capture_screen(self, camera_frame: np.ndarray) -> np.ndarray:
        display_frame = camera_frame.copy()

        self._draw_stage_indicator(display_frame, 2, 3)

        instructions = [
            "Look directly at the camera",
            "Press SPACE to capture",
            "Press Q to quit"
        ]
        self._draw_instructions(display_frame, instructions, 600)

        return display_frame

    def _render_processing_screen(self, message: str) -> np.ndarray:
        frame = self._create_blank_frame()
        self._draw_text_centered(
            frame,
            message,
            self._config.window_height // 2,
            self._config.colors.white
        )
        return frame

    def _render_result_screen(
        self,
        result: VerificationResult
    ) -> np.ndarray:
        frame = self._create_blank_frame()

        self._draw_stage_indicator(frame, 3, 3)

        if result.is_match:
            status_text = "VERIFIED"
            status_color = self._config.colors.green
        else:
            status_text = "REJECTED"
            status_color = self._config.colors.red

        self._draw_text_centered(
            frame,
            status_text,
            self._config.window_height // 2 - 50,
            status_color
        )

        similarity_text = f"Similarity: {result.similarity_score:.2f}"
        self._draw_text_centered(
            frame,
            similarity_text,
            self._config.window_height // 2 + 20,
            self._config.colors.white
        )

        threshold_text = f"Threshold: {result.threshold:.2f}"
        self._draw_text_centered(
            frame,
            threshold_text,
            self._config.window_height // 2 + 70,
            self._config.colors.white
        )

        instructions = ["Press R to restart", "Press Q to quit"]
        self._draw_instructions(frame, instructions, 600)

        return frame

    def _render_error_screen(self, error_message: str) -> np.ndarray:
        frame = self._create_blank_frame()

        self._draw_text_centered(
            frame,
            "ERROR",
            self._config.window_height // 2 - 50,
            self._config.colors.red
        )

        self._draw_text_centered(
            frame,
            error_message,
            self._config.window_height // 2 + 20,
            self._config.colors.white
        )

        instructions = ["Press R to retry", "Press Q to quit"]
        self._draw_instructions(frame, instructions, 600)

        return frame

    def show_card_capture(self) -> Optional[np.ndarray]:
        self._ui_state = UIState.CARD_CAPTURE

        if not self._camera.isOpened():
            return None

        ret, frame = self._camera.read()
        if not ret:
            return None

        display_frame = self._render_card_capture_screen(frame)
        cv2.imshow(self._config.window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            return frame
        if key == ord('q'):
            return None

        return self.show_card_capture()

    def show_live_capture(self) -> Optional[np.ndarray]:
        self._ui_state = UIState.LIVE_CAPTURE

        if not self._camera.isOpened():
            return None

        ret, frame = self._camera.read()
        if not ret:
            return None

        display_frame = self._render_live_capture_screen(frame)
        cv2.imshow(self._config.window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            return frame
        if key == ord('q'):
            return None

        return self.show_live_capture()

    def show_processing(self, message: str) -> None:
        self._ui_state = UIState.CARD_PROCESSING
        frame = self._render_processing_screen(message)
        cv2.imshow(self._config.window_name, frame)
        cv2.waitKey(1)

    def show_result(self, result: VerificationResult) -> str:
        self._ui_state = UIState.RESULT_DISPLAY
        frame = self._render_result_screen(result)
        cv2.imshow(self._config.window_name, frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                return 'restart'
            if key == ord('q'):
                return 'quit'

    def show_error(self, error_message: str) -> str:
        self._ui_state = UIState.ERROR_DISPLAY
        self._error_message = error_message

        frame = self._render_error_screen(error_message)
        cv2.imshow(self._config.window_name, frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                return 'retry'
            if key == ord('q'):
                return 'quit'

    def initialize(self) -> bool:
        success = self._initialize_camera()
        if success:
            cv2.namedWindow(self._config.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self._config.window_name,
                self._config.window_width,
                self._config.window_height
            )
        return success

    def cleanup(self) -> None:
        self._release_camera()
        cv2.destroyAllWindows()

def create_verification_ui(config: Optional[UIConfig] = None) -> VerificationUI:
    actual_config = config if config else UIConfig()
    return VerificationUI(actual_config)
