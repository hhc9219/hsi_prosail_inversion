"""
A script to accurately convert a ENVI hyperspectral image to a sRGB image.
By default this script uses a D65 illuminant and the CIE 1931 2 Degree Standard Observer CMF.

hhc9219@rit.edu
"""


def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    import numpy as np
    from pathlib import Path
    from modules import hsi_io
    from modules.hsi_to_rgb_conversion import hsi_to_sRGB_mp

    img_name = input("Please enter the hsi to process from hsi_config.json: ")

    hsi_hdr_path = Path(HSI_CONFIG[img_name]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[img_name]["img"])

    wavelengths = hsi_io.get_wavelengths(hsi_hdr_path)

    hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    h, w, _ = hsi.shape
    with hsi_io.Memmap(OUTPUT_FOLDER / "rgb_result.npy", shape=(h, w, 3), dtype=np.uint8, mode="w+") as rgb_result:
        if rgb_result.array is not None:

            hsi_to_sRGB_mp(
                hsi_src=hsi,
                rgb_dst=rgb_result.array,
                original_wavelengths=wavelengths,
                num_threads=8,
                max_bytes=int(1e9),
            )

    del hsi


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()