import numpy as np
from pathlib import Path
from typing import Any
from tqdm import tqdm


class Memmap:
    def __init__(
        self,
        npy_path: Path,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float64,
        mode: str = "r",
        max_bytes: int | None = None,
        show_progress: bool = False,
    ):
        self.npy_path = npy_path
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.max_bytes = max_bytes
        self.show_progress = show_progress
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

    def __getitem__(self, key: slice | int):
        if self.array is None:
            raise RuntimeError("Memmap array is not open.")
        return self.array[key]

    def __setitem__(self, key, value: "np.ndarray | Memmap"):
        if self.array is None:
            raise RuntimeError("Memmap array is not open.")
        if self.mode not in ["c", "r+", "w+", "a+"]:
            raise ValueError("Mode must be 'c', 'r+', 'w+', or 'a+' for assignment.")
        if isinstance(value, np.ndarray):
            if self.max_bytes is not None and value.nbytes > self.max_bytes:
                if self.mode == "c":
                    raise ValueError("Cannot exceed max_bytes on assignment while in mode 'c'.")
                if value.itemsize != self.array.itemsize:
                    raise TypeError(
                        f"Itemsize does not match, potential dtype mismatch.\nDestination: {self.array.dtype}\nSource: {value.dtype}"
                    )
                max_elements_per_part = self.max_bytes // self.array.itemsize
                num_parts = -(-value.size // max_elements_per_part)
                start = 0
                if self.show_progress:
                    progress_bar = tqdm(total=value.size, desc="Copying Elements")
                src = value.flatten()
                self.close_array()
                self_mode = self.mode
                self.mode = "r+"
                try:
                    for _ in range(num_parts - 1):
                        end = start + max_elements_per_part
                        with self:
                            self.array[key].flatten()[start:end] = src[start:end]
                        if self.show_progress:
                            progress_bar.update(end - start)
                        start = end
                    with self:
                        self.array[key].flatten()[start:] = src[start:]
                finally:
                    self.mode = self_mode
                    self.open_array()
                    if self.show_progress:
                        progress_bar.update(progress_bar.total - progress_bar.n)
                        progress_bar.close()
            else:
                self.array[key] = value
        elif isinstance(value, Memmap):
            if value.array is None:
                raise RuntimeError("Other Memmap array is not open.")
            if value.mode not in ["c", "r+", "w+", "a+"]:
                raise ValueError("Other Memmap's Mode must be 'c', 'r+', 'w+', or 'a+' for assignment.")
            if self.max_bytes is not None and value.max_bytes is not None:
                max_bytes = min(self.max_bytes, value.max_bytes)
            elif self.max_bytes is not None and value.max_bytes is None:
                max_bytes = self.max_bytes
            elif self.max_bytes is None and value.max_bytes is not None:
                max_bytes = value.max_bytes
            else:
                max_bytes = None
            if max_bytes is not None and value.array.nbytes > max_bytes:
                if self.mode == "c":
                    raise ValueError("Cannot exceed max_bytes on assignment while in mode 'c'.")
                if value.array.itemsize != self.array.itemsize:
                    raise TypeError(
                        f"Itemsize does not match, potential dtype mismatch.\nDestination: {self.array.dtype}\nSource: {value.array.dtype}"
                    )
                max_elements_per_part = max_bytes // self.array.itemsize
                num_parts = -(-value.array.size // max_elements_per_part)
                start = 0
                if self.show_progress:
                    progress_bar = tqdm(total=value.array.size, desc="Copying Elements")
                self.close_array()
                value.close_array()
                self_mode = self.mode
                value_mode = value.mode
                self.mode = "r+"
                value.mode = "r"
                try:
                    for _ in range(num_parts - 1):
                        end = start + max_elements_per_part
                        with self:
                            with value:
                                self.array[key].flatten()[start:end] = value.array.flatten()[start:end]
                        if self.show_progress:
                            progress_bar.update(end - start)
                        start = end
                    with self:
                        with value:
                            self.array[key].flatten()[start:] = value.array.flatten()[start:]
                finally:
                    self.mode = self_mode
                    value.mode = value_mode
                    self.open_array()
                    value.open_array()
                    if self.show_progress:
                        progress_bar.update(progress_bar.total - progress_bar.n)
                        progress_bar.close()
            else:
                self.array[key] = value.array
