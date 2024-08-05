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
    from modules.prosail_inversion import invert_prosail_mp, calculate_ndvi_mp, where_dark_mp

    # ================================================================================================================
    # SETUP
    # ----------------------------------------------------------------------------------------------------------------

    NUM_THREADS = max(THREADS - 3, 3)
    MIN_BYTES = int(0.05e9)

    WHERE_DARK_MEM_FACTOR = 1 / 6
    NDVI_MEM_FACTOR = 1 / 8
    PROSAIL_MEM_FACTOR = 1 / 160

    HSI_NAME = input("Please enter the hsi to process from hsi_config.json: ")

    DARK_THRESHOLD = 1e-9
    NDVI_THRESHOLD = 0.1

    ATOL_RMSE_RESIDUAL = 0.1  # 0.1-0.01
    ATOL_WAVELENGTH = 10  # 10-1
    MAXITER_FACTOR = 100  # 100-200
    IS_ADAPTIVE = True
    PRINT_ERRORS = True

    hsi_hdr_path = Path(HSI_CONFIG[HSI_NAME]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[HSI_NAME]["img"])

    wavelengths = hsi_io.get_wavelengths(hsi_hdr_path)

    hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    h, w, _ = hsi.shape

    # ================================================================================================================
    # WHERE DARK MASK
    # ----------------------------------------------------------------------------------------------------------------

    dark_path = OUTPUT_FOLDER / f"{HSI_NAME}_dark.npy"
    if (
        dark_path.exists()
        and input(f"Would you like to use the last found dark pixels for: {HSI_NAME} ?\n(y/n): ") == "y"
    ):
        where_dark = np.load(dark_path)
    else:
        print("\nFinding dark pixels:")
        where_dark = np.empty(shape=(h, w, 1), dtype=bool)
        where_dark_mp(
            hsi=hsi,
            where_dark_dst=where_dark,
            num_threads=NUM_THREADS,
            max_bytes=max(int(MEMORY * WHERE_DARK_MEM_FACTOR * 1e9), MIN_BYTES),
            dark_threshold=DARK_THRESHOLD,
            show_progress=True,
        )
        # Free memory
        del hsi
        hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
        print("Found dark pixels.")
        np.save(file=dark_path, arr=where_dark, allow_pickle=False, fix_imports=False)
        print(f"The dark pixels were saved to the ouput folder as:\n{dark_path.name}\n")

    # ================================================================================================================
    # NDVI
    # ----------------------------------------------------------------------------------------------------------------

    ndvi_path = OUTPUT_FOLDER / f"{HSI_NAME}_ndvi.npy"
    if (
        ndvi_path.exists()
        and input(f"Would you like to use the last calculated NDVI for: {HSI_NAME} ?\n(y/n): ") == "y"
    ):
        ndvi = np.load(ndvi_path)
        ndvi[where_dark] = 0
    else:
        print("\nCalculating NDVI:")
        ndvi = np.empty(shape=(h, w, 1), dtype=np.float64)
        calculate_ndvi_mp(
            hsi=hsi,
            ndvi_dst=ndvi,
            wavelengths=wavelengths,
            num_threads=NUM_THREADS,
            max_bytes=max(int(MEMORY * NDVI_MEM_FACTOR * 1e9), MIN_BYTES),
            zero_threshold=1e-9,
            show_progress=True,
        )
        # Free memory
        del hsi
        hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
        print("NDVI calculation complete.")
        ndvi[where_dark] = 0
        np.save(file=ndvi_path, arr=ndvi, allow_pickle=False, fix_imports=False)
        print(f"The NDVI was saved to the ouput folder as:\n{ndvi_path.name}\n")

    # ================================================================================================================
    # GEOMETRY
    # ----------------------------------------------------------------------------------------------------------------

    # ================================================================================================================
    # MASK CREATION
    # ----------------------------------------------------------------------------------------------------------------

    where_ndvi = ndvi > NDVI_THRESHOLD
    mask = where_ndvi & ~where_dark

    # ================================================================================================================
    # PROSAIL
    # ----------------------------------------------------------------------------------------------------------------

    print("\nInverting PROSAIL:")
    inv_path = OUTPUT_FOLDER / f"{HSI_NAME}_inv_res.npy"
    with hsi_io.Memmap(inv_path, shape=(h, w, 8), dtype=np.float64, mode="w+") as inversion_result:
        if inversion_result.array is not None:
            invert_prosail_mp(
                hsi_geo_mask_stack_src=hsi,
                inversion_result_dst=inversion_result.array,
                wavelengths=wavelengths,
                atol_rmse_residual=ATOL_RMSE_RESIDUAL,
                atol_wavelength=ATOL_WAVELENGTH,
                maxiter_factor=MAXITER_FACTOR,
                is_adaptive=IS_ADAPTIVE,
                print_errors=PRINT_ERRORS,
                num_threads=NUM_THREADS,
                max_bytes=max(int(MEMORY * PROSAIL_MEM_FACTOR * 1e9), MIN_BYTES),
                show_progress=True,
            )
    del hsi
    print("PROSAIL inversion complete.")
    print(f"The PROSAIL inversion result was saved to the output folder as:\n{inv_path.name}\n")

    # ================================================================================================================
    # END
    # ----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()
