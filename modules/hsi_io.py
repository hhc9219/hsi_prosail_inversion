"""
This file defines multiple routines for loading data from ENVI hyperspectral images.

It includes functionality for:
- Selecting files using a GUI file dialog.
- Locating corresponding hyperspectral image data files based on header files.
- Opening hyperspectral image data as numpy memory-mapped arrays for efficient access and manipulation.
- Parsing wavelength information from header files.
- Extracting ancillary data such as longitude, latitude, sensor zenith, sensor azimuth, solar zenith, and solar azimuth.

The provided routines are designed to facilitate the handling and processing of hyperspectral image data, which is 
commonly used in remote sensing.

hhc9219@rit.edu
"""

import os
import numpy as np
import spectral.io.envi as envi  # type:ignore
from pathlib import Path
from typing import Any
from numpy.typing import NDArray

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()


class ParseEnviError(Exception):
    pass


def open_file_path(folder_name: str | None = None, ext: str | None = None, **kwargs: Any):
    initialdir = kwargs["initialdir"] if "initialdir" in kwargs else os.getcwd()
    if folder_name:
        folder_path = Path(initialdir) / folder_name
        if folder_path.exists():
            initialdir = str(folder_path)
    kwargs["initialdir"] = initialdir
    if ext:
        filetypes = [(f"{ext[1:].upper()} File", f"*{ext}")]
    else:
        filetypes = kwargs["filetypes"] if "filetypes" in kwargs else None
    if filetypes:
        kwargs["filetypes"] = filetypes
    file_path = Path(filedialog.askopenfilename(**kwargs))
    if str(file_path) == ".":
        raise Exception("No file was selected.")
    return file_path


def get_envi_hsi_data_path(img_hdr_path: Path, ext: str | list[str] = ["", ".img", ".dat"]):
    assert img_hdr_path.exists()
    assert img_hdr_path.suffix == ".hdr"
    name = img_hdr_path.stem
    folder = img_hdr_path.parent
    if isinstance(ext, str):
        ext = [ext]
    for e in ext:
        hsi_path = folder / (name + e)
        if hsi_path.exists():
            return hsi_path
    raise FileNotFoundError("Could not find hsi data location.")


def open_envi_hsi_as_np_memmap(
    img_hdr_path: Path | None = None, img_data_path: Path | None = None, writable: bool | None = False
):
    if not img_hdr_path:
        img_hdr_path = open_file_path(folder_name="hsi", ext=".hdr")

    if img_data_path:
        assert img_hdr_path.exists()
    else:
        img_data_path = get_envi_hsi_data_path(img_hdr_path)

    spy_img: Any = envi.open(img_hdr_path, img_data_path)  # type:ignore
    np_img: NDArray = spy_img.open_memmap(writable=writable)
    return np_img


def get_wavelengths(img_hdr_path: Path):
    """
    Parses wavelengths from a .hdr file.
    """
    assert img_hdr_path.exists()
    wavelengths_str = ""
    looking_at_wavelengths = False
    hdr_file = open(img_hdr_path, "r")
    try:
        for line in hdr_file:
            if looking_at_wavelengths:
                wavelengths_str += line
                if line.rstrip().endswith("}"):
                    break
            else:
                if line.replace(" ", "").startswith("wavelength={"):
                    looking_at_wavelengths = True
                    wavelengths_str += line
                    if line.rstrip().endswith("}"):
                        break
        hdr_file.close()
        wavelengths_str_list = wavelengths_str.replace("\n", "").replace(" ", "")[12:-1].split(",")
        wavelengths_array = np.empty(len(wavelengths_str_list), dtype=np.float64)
        for i, v in enumerate(wavelengths_str_list):
            wavelengths_array[i] = np.float64(v)
        return wavelengths_array
    except Exception:
        hdr_file.close()

        raise ParseEnviError("Failed to parse wavelengths from hyperspectral image.")


def get_anc_data(anc_hdr_path: Path, anc_data_path: Path):
    anc = open_envi_hsi_as_np_memmap(anc_hdr_path, anc_data_path)
    longitude = anc[:, :, 0]
    latitude = anc[:, :, 1]
    sensor_zenith = anc[:, :, 2]
    sensor_azimuth = anc[:, :, 3]
    solar_zenith = anc[:, :, 4]
    solar_azimuth = anc[:, :, 5]
    return longitude, latitude, sensor_zenith, sensor_azimuth, solar_zenith, solar_azimuth


class Memmap:
    def __init__(
        self, memmap_dat_path: Path, dtype: type = np.float64, mode: str = "r", shape: tuple[int, ...] | None = None
    ):
        self.dat_path = memmap_dat_path
        self.shape = shape
        self.mode = mode
        self.dtype = dtype
        self.array = None

    def open_array(self):
        if self.mode == "r" or self.mode == "w+" or self.mode == "r+" or self.mode == "c":
            self.array = np.memmap(self.dat_path, dtype=self.dtype, mode=self.mode, shape=self.shape)
        else:
            raise ValueError("Mode must be r or w+")
        return self.array

    def close_array(self):
        if self.array is not None:
            if self.mode == "w+" or self.mode == "r+":
                self.array.flush()
            del self.array
            self.array = None

    def __enter__(self):
        self.open_array()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_array()
