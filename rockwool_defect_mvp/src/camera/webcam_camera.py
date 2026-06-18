from __future__ import annotations

import cv2
import numpy as np

from src.camera.camera_base import CameraBase


class WebcamCamera(CameraBase):
    """Captures frames from a webcam using OpenCV VideoCapture."""

    def __init__(self, webcam_index: int = 0, width: int | None = None, height: int | None = None) -> None:
        self.webcam_index = webcam_index
        self._capture = self._open_capture(webcam_index)

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open webcam with index: {webcam_index}")

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def get_frame(self) -> np.ndarray:
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not read frame from webcam index: {self.webcam_index}")
        return frame

    def release(self) -> None:
        if self._capture.isOpened():
            self._capture.release()

    def _open_capture(self, webcam_index: int) -> cv2.VideoCapture:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(webcam_index, backend)
            if capture.isOpened():
                return capture
            capture.release()

        return cv2.VideoCapture(webcam_index)
