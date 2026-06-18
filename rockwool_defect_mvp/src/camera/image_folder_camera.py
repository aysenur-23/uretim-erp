from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.camera.camera_base import CameraBase


class ImageFolderCamera(CameraBase):
    """Reads image files from a folder one by one."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, folder_path: str | Path, loop: bool = True) -> None:
        self.folder_path = Path(folder_path)
        self.loop = loop
        self._image_paths = self._load_image_paths()
        self._index = 0

    def get_frame(self) -> np.ndarray:
        if not self._image_paths:
            raise RuntimeError(f"No jpg/jpeg/png images found in: {self.folder_path}")

        attempts = 0
        while attempts < len(self._image_paths):
            image_path = self._image_paths[self._index]
            self._advance_index()

            frame = cv2.imread(str(image_path))
            if frame is not None:
                return frame

            attempts += 1

        raise RuntimeError(f"Could not read any image from: {self.folder_path}")

    def _load_image_paths(self) -> list[Path]:
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Image folder does not exist: {self.folder_path}")

        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Image folder path is not a directory: {self.folder_path}")

        return sorted(
            path
            for path in self.folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def _advance_index(self) -> None:
        self._index += 1
        if self._index >= len(self._image_paths):
            self._index = 0 if self.loop else len(self._image_paths) - 1
