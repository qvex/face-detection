from pathlib import Path
from typing import Optional
import questionary

from src.core.types import Result, Success, Failure
from src.core.errors import DetectionError, DetectionErrorKind

class ImageBrowser:
    def __init__(self, root_dir: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
        self._root_dir = root_dir
        self._extensions = extensions

    def _find_images(self) -> list[Path]:
        images = []
        for ext in self._extensions:
            images.extend(self._root_dir.rglob(f'*{ext}'))
            images.extend(self._root_dir.rglob(f'*{ext.upper()}'))
        return sorted(images)

    def _format_choice(self, image_path: Path) -> str:
        relative = image_path.relative_to(self._root_dir)
        return str(relative)

    def select_image(self) -> Result[Path, DetectionError]:
        if not self._root_dir.exists():
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"directory not found: {self._root_dir}"
            ))

        images = self._find_images()

        if len(images) == 0:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"no images found in {self._root_dir}"
            ))

        choices = [self._format_choice(img) for img in images]

        print()
        print(f"Found {len(images)} images in {self._root_dir}")
        print()

        selected = questionary.select(
            "Select reference image (use arrow keys, press Enter to confirm):",
            choices=choices,
            use_shortcuts=False,
            use_indicator=True
        ).ask()

        if selected is None:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details="selection cancelled by user"
            ))

        selected_index = choices.index(selected)
        selected_path = images[selected_index]

        return Success(selected_path)

def create_image_browser(
    root_dir: Optional[Path] = None,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> ImageBrowser:
    default_root = Path("data/test")
    actual_root = root_dir if root_dir else default_root
    return ImageBrowser(actual_root, extensions)
