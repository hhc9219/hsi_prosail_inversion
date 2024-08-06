"""
A script to accurately convert a ENVI hyperspectral image to a sRGB image.
By default this script uses a D65 illuminant and the CIE 1931 2 Degree Standard Observer CMFs.

hhc9219@rit.edu
"""


def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    import numpy as np
    from pathlib import Path
    from modules.npmemmap import Memmap
    from modules.hsi_io import get_wavelengths, open_envi_hsi_as_np_memmap
    from modules.hsi_to_rgb_conversion import hsi_to_sRGB_mp

    img_name = input("Please enter the hsi to process from hsi_config.json: ")

    num_threads = max(THREADS - 3, 3)
    min_bytes = int(0.05e9)
    max_bytes = max(int(MEMORY / 64 * 1e9), min_bytes)

    hsi_hdr_path = Path(HSI_CONFIG[img_name]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[img_name]["img"])

    wavelengths = get_wavelengths(hsi_hdr_path)

    hsi = open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    h, w, _ = hsi.shape
    rgb_path = OUTPUT_FOLDER / f"{img_name}_rgb_res.npy"
    with Memmap(rgb_path, shape=(h, w, 3), dtype=np.uint8, mode="w+") as rgb_result:
        if rgb_result.array is not None:

            hsi_to_sRGB_mp(
                hsi_src=hsi,
                rgb_dst=rgb_result.array,
                original_wavelengths=wavelengths,
                num_threads=num_threads,
                max_bytes=max_bytes,
            )

    del hsi

    if input("Would you like to convert the output .npy file to a .png? (y/n): ") == "y":
        from matplotlib import pyplot as plt

        rgb = np.load(rgb_path)
        plt.imsave(OUTPUT_FOLDER / f"{img_name}_rgb_res.png", rgb)

    print("HSI to RGB conversion is complete.")


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()
