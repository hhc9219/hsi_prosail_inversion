"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""


def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    import numpy as np
    from pathlib import Path
    from modules import hsi_io
    from modules.prosail_inversion import invert_prosail_mp

    img_name = input("Please enter the hsi to process from hsi_config.json: ")

    num_threads = max(THREADS - 3, 3)
    max_bytes = max(int(MEMORY / 64 * 1e9), int(0.05e9))

    hsi_hdr_path = Path(HSI_CONFIG[img_name]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[img_name]["img"])

    wavelengths = hsi_io.get_wavelengths(hsi_hdr_path)

    hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    h, w, _ = hsi.shape
    with hsi_io.Memmap(
        OUTPUT_FOLDER / "inversion_result.npy", shape=(h, w, 11), dtype=np.float64, mode="w+"
    ) as inversion_result:
        if inversion_result.array is not None:

            invert_prosail_mp(
                hsi_src=hsi,
                inversion_result_dst=inversion_result.array,
                wavelengths=wavelengths,
                atol_rmse_residual=0.05,
                atol_wavelength=5,
                maxiter_factor=200,
                ndvi_threshold=0.1,
                black_threshold=1e-9,
                is_adaptive=True,
                num_threads=num_threads,
                max_bytes=max_bytes,
                print_errors=False,
            )

    del hsi


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()
