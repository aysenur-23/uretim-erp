from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraBase(ABC):
    """Common interface for all image acquisition sources."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return the next BGR frame as a NumPy array."""
        raise NotImplementedError

    def release(self) -> None:
        """Release camera resources when the source owns any."""
        return None
