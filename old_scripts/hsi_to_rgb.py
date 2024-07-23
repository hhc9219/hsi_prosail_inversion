"""
Old script to convert a specific hyperspectral image to an sRGB image.
Was in the project root directory.

hhc9219@rit.edu
"""

from modules.context_manager import Context, enforce_venv

enforce_venv(__file__)

# Imports
import numpy as np
from pathlib import Path
from modules import hsi_io
from modules.hsi_processing import hsi_to_sRGB_mp


def main():
    threads, memory = get_resource_values()
    hsi_config = get_hsi_config()
    project_folder = get_project_folder()
    output_folder = project_folder / "output"

    uas_hsi_small_hdr = Path(hsi_config["uas_hsi_small"]["hdr"])
    # uas_hsi_small_img = Path(hsi_config["uas_hsi_small"]["img"])
    uas_hsi_large_hdr = Path(hsi_config["uas_hsi_large"]["hdr"])
    uas_hsi_large_img = Path(hsi_config["uas_hsi_large"]["img"])

    uas_hsi_small_wavelengths = hsi_io.get_wavelengths(uas_hsi_small_hdr)
    uas_hsi_large_wavelengths = hsi_io.get_wavelengths(uas_hsi_large_hdr)
    assert np.allclose(uas_hsi_small_wavelengths, uas_hsi_large_wavelengths)
    wavelengths = uas_hsi_small_wavelengths

    uas_hsi_large = hsi_io.open_envi_hsi_as_np_memmap(
        img_hdr_path=uas_hsi_large_hdr, img_data_path=uas_hsi_large_img, writable=False
    )
    h, w, _ = uas_hsi_large.shape
    with hsi_io.Memmap(
        output_folder / "uas_rgb_large.npy", shape=(h, w, 3), dtype=np.uint8, mode="w+"
    ) as uas_rgb_large:
        if uas_rgb_large.array is not None:
            hsi_to_sRGB_mp(
                uas_hsi_large,
                uas_rgb_large.array,
                original_wavelengths=wavelengths,
                num_threads=8,
                max_bytes=int(0.3e9),
            )
    del uas_hsi_large


def get_project_folder():
    with Context() as project:
        project_folder = project.context_folder
    return project_folder


def get_resource_values() -> tuple[int, int | float]:
    with Context(data_filename="resource_config.json") as resource_config:
        if resource_config.data:
            resource_values = resource_config.data["cpu_thread_count"], resource_config.data["memory_GB"]
        else:
            raise FileNotFoundError("resource_config.json was not found")
    return resource_values


def get_hsi_config():
    with Context(data_filename="hsi_config.json") as hsi_config:
        if hsi_config.data:
            hsi_config_info = hsi_config.data
        else:
            raise FileNotFoundError("hsi_config.json was not found")
    return hsi_config_info


if __name__ == "__main__":
    main()
