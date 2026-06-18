from __future__ import annotations

import cv2
import numpy as np

from src.camera.camera_base import CameraBase


class UploadedImageCamera(CameraBase):
    """Uses a browser-uploaded image as a single-frame source."""

    def __init__(self, image_bytes: bytes) -> None:
        self._frame = self._decode_image(image_bytes)

    def get_frame(self) -> np.ndarray:
        return self._frame.copy()

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ValueError("Uploaded image is empty.")

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Uploaded file could not be decoded as an image.")

        return frame
