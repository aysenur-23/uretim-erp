from __future__ import annotations

import cv2
import numpy as np


def resize_image(image: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """Resize an image while preserving aspect ratio."""
    if image.size == 0:
        raise ValueError("Cannot resize an empty image.")

    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()

    scale = max_width / float(width)
    new_size = (max_width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def convert_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur_image(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Reduce small texture noise with Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance local contrast using CLAHE."""
    gray = convert_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """Reduce uneven illumination in a grayscale image."""
    gray = convert_gray(image)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized
