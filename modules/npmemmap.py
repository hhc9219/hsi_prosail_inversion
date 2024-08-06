import numpy as np
from pathlib import Path
from typing import Any


class Memmap:
    def __init__(
        self,
        npy_path: Path,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float64,
        mode: str = "r",
    ):
        self.npy_path = npy_path
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.array: np.memmap[Any, Any] | None = None
        self.is_open = False

    def open_array(self):
        if self.mode not in ["r", "c", "r+", "w+", "a+"]:
            raise ValueError("Mode must be 'r', 'c', 'r+', 'w+', or 'a+'.")
        mode = self.mode if self.mode != "a+" else "r+" if self.npy_path.exists() else "w+"
        if mode == "w+" and self.shape is None:
            raise ValueError("Shape must be specified when creating a new file.")
        self.is_open = True
        self.array = np.lib.format.open_memmap(self.npy_path, mode=mode, dtype=self.dtype, shape=self.shape)

    def close_array(self):
        if self.array is not None:
            if self.mode in ["r+", "w+", "a+"]:
                self.array.flush()
            del self.array
            self.array = None
        self.is_open = False

    def __enter__(self):
        self.open_array()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_array()
