"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""

from modules.context_manager import Context, enforce_venv

enforce_venv(__file__)

# Imports
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Manager
from modules import hsi_io, prosail_data


def main():
    resource_data = get_resource_config()
    hsi_data = get_hsi_config()

    uas_hsi_small_hdr = Path(hsi_data["uas_hsi_small"]["hdr"])
    uas_hsi_small_img = Path(hsi_data["uas_hsi_small"]["img"])
    uas_hsi_large_hdr = Path(hsi_data["uas_hsi_large"]["hdr"])
    uas_hsi_large_img = Path(hsi_data["uas_hsi_large"]["img"])

    uas_hsi_small_wavelengths = hsi_io.get_wavelengths(uas_hsi_small_hdr)
    uas_hsi_large_wavelengths = hsi_io.get_wavelengths(uas_hsi_large_hdr)
    assert np.allclose(uas_hsi_small_wavelengths, uas_hsi_large_wavelengths)
    wavelengths = uas_hsi_small_wavelengths

    uas_hsi_small = hsi_io.open_envi_hsi_as_np_memmap(uas_hsi_small_hdr, uas_hsi_small_img)
    uas_hsi_large = hsi_io.open_envi_hsi_as_np_memmap(uas_hsi_large_hdr, uas_hsi_large_img)


def get_resource_config():
    with Context(data_filename="resource_config.json") as resource_config:
        if resource_config.data:
            return resource_config.data
        else:
            raise FileNotFoundError("resource_config.json was not found")

def get_hsi_config():
    with Context(data_filename="hsi_config.json") as hsi_config:
        if hsi_config.data:
            return hsi_config.data
        else:
            raise FileNotFoundError("hsi_config.json was not found")

if __name__ == "__main__":
    main()
