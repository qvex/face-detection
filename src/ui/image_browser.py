from pathlib import Path
from typing import Optional
import questionary

from src.core.types import Result, Success, Failure
from src.core.errors import DetectionError, DetectionErrorKind

class ImageBrowser:
    def __init__(self, root_dir: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
        self._root_dir = root_dir
        self._extensions = extensions

    def _is_image(self, path: Path) -> bool:
        return path.suffix.lower() in self._extensions

    def _get_directory_contents(self, directory: Path) -> tuple[list[Path], list[Path]]:
        folders = []
        images = []

        for item in sorted(directory.iterdir()):
            if item.is_dir():
                folders.append(item)
            elif item.is_file() and self._is_image(item):
                images.append(item)

        return folders, images

    def _format_folder(self, folder: Path) -> str:
        return f"[DIR] {folder.name}/"

    def _format_image(self, image: Path) -> str:
        return f"      {image.name}"

    def _browse_directory(self, current_dir: Path) -> Result[Path, DetectionError]:
        folders, images = self._get_directory_contents(current_dir)

        choices = []
        items = []

        if current_dir != self._root_dir:
            choices.append("[DIR] ../  (Go back)")
            items.append("back")

        for folder in folders:
            choices.append(self._format_folder(folder))
            items.append(folder)

        for image in images:
            choices.append(self._format_image(image))
            items.append(image)

        if len(choices) == 0:
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"no images or folders found in {current_dir}"
            ))

        relative = current_dir.relative_to(self._root_dir) if current_dir != self._root_dir else Path(".")
        location = f"Location: {self._root_dir.name}/{relative}" if str(relative) != "." else f"Location: {self._root_dir.name}/"

        print()
        print(location)
        print(f"Folders: {len(folders)}, Images: {len(images)}")
        print()

        selected = questionary.select(
            "Select image or folder (use arrow keys, Enter to confirm, Ctrl+C to cancel):",
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
        selected_item = items[selected_index]

        if selected_item == "back":
            return self._browse_directory(current_dir.parent)
        elif isinstance(selected_item, Path) and selected_item.is_dir():
            return self._browse_directory(selected_item)
        else:
            return Success(selected_item)

    def select_image(self) -> Result[Path, DetectionError]:
        if not self._root_dir.exists():
            return Failure(DetectionError(
                kind=DetectionErrorKind.INVALID_IMAGE,
                details=f"directory not found: {self._root_dir}"
            ))

        return self._browse_directory(self._root_dir)

def create_image_browser(
    root_dir: Optional[Path] = None,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> ImageBrowser:
    default_root = Path("data/test")
    actual_root = root_dir if root_dir else default_root
    return ImageBrowser(actual_root, extensions)
