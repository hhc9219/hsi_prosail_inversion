"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""


def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    from pathlib import Path
    import sys
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from modules.npmemmap import Memmap
    from modules.hsi_io import open_envi_hsi_as_np_memmap, get_wavelengths
    from modules.hsi_processing import where_dark_mp, calculate_ndvi_mp, copy_add_channels_mp
    from modules.prosail_inversion import invert_prosail_mp

    # ================================================================================================================
    # SETUP
    # ----------------------------------------------------------------------------------------------------------------

    # Select hsi and path locations
    HSI_NAME = input("Please enter the hsi to process from hsi_config.json: ")
    hsi_hdr_path = Path(HSI_CONFIG[HSI_NAME]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[HSI_NAME]["img"])

    # PROSAIL inversion call settings
    # See modules.prosail_inversion for documentation
    # (For specific parameter settings: See modules.prosail_data)
    MIN_WAVELENGTH = 400.0  # minimally 400 [nm]
    MAX_WAVELENGTH = 900.0  # maximally 2500 [nm]
    ATOL_RMSE_RESIDUAL = 0.01  # 0.1-0.01
    ATOL_WAVELENGTH = 1  # 10-1
    MAXITER_FACTOR = 200  # 100-200
    IS_ADAPTIVE = True
    PRINT_ERRORS = True
    # end

    # Masking threshold settings
    DARK_THRESHOLD = 1e-9
    NDVI_THRESHOLD = 0.1
    NDVI_BLUR_SIGMA = -1.0  # -1.0 for no blur, > 0 for blur
    MIN_TO_INVERT_PCT = 1e-9  # 0-1 : The minimum percentage of pixels which should be inverted
    # end

    # Memory management settings
    NUM_THREADS = max(THREADS - 3, 3)
    MIN_BYTES = int(0.05e9)
    WHERE_DARK_MEM_FACTOR = 0.5
    NDVI_MEM_FACTOR = 0.5
    HGMS_MEM_FACTOR = 0.35
    PROSAIL_MEM_FACTOR = 0.35
    # end

    # informative settings
    GEO_GOOGLE_DRIVE_LINK = "https://drive.google.com/drive/folders/1zkj31M0oDlEoTi-1RGWNE6L7nLUpwGm1?usp=sharing"
    GEO_CIS_LINUX_PATH = "Not uploaded yet."
    # end

    # visualization settings
    GEO_PNG_MISSING_PCT_DARKER = 0.2  # 0-1 : The percentage darker missing solar and sensor geometry should display
    # end

    IMG_OUTPUT_FOLDER = OUTPUT_FOLDER / "images"
    IMG_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Load hsi wavelengths, height, width, depth (channels)
    wavelengths = get_wavelengths(hsi_hdr_path)
    temp_hsi = open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    try:
        h, w, d = temp_hsi.shape
        dt = temp_hsi.dtype
    finally:
        del temp_hsi

    # ================================================================================================================
    # WHERE DARK MASK
    # ----------------------------------------------------------------------------------------------------------------

    dark_path = OUTPUT_FOLDER / f"{HSI_NAME}_dark.npy"
    if (
        not dark_path.exists()
        or input(f"Would you like to use the last found dark pixels for: {HSI_NAME} ? (y/n): ").strip().lower() != "y"
    ):
        print("\nFinding dark pixels:")
        where_dark_mp(
            src_hsi_hdr_path=hsi_hdr_path,
            src_hsi_data_path=hsi_img_path,
            where_dark_dst_npy_path=dark_path,
            num_threads=NUM_THREADS,
            max_bytes=max(int(MEMORY * WHERE_DARK_MEM_FACTOR * 1e9), MIN_BYTES),
            dark_threshold=DARK_THRESHOLD,
            show_progress=True,
        )
        print("Found dark pixels.")
        print(f"The dark pixels were saved to the ouput folder as:\n{dark_path.name}\n")

    where_dark = np.load(dark_path)

    where_dark_mask_img = Image.fromarray(where_dark[:, :, 0].astype(np.uint8) * 255, mode="L")
    where_dark_mask_img.save(IMG_OUTPUT_FOLDER / f"{HSI_NAME}_dark.png")
    del where_dark_mask_img

    # ================================================================================================================
    # NDVI
    # ----------------------------------------------------------------------------------------------------------------

    ndvi_path = OUTPUT_FOLDER / f"{HSI_NAME}_ndvi.npy"
    if (
        not ndvi_path.exists()
        or input(f"Would you like to use the last calculated NDVI for: {HSI_NAME} ? (y/n): ").strip().lower() != "y"
    ):
        print("\nCalculating NDVI:")
        calculate_ndvi_mp(
            src_hsi_hdr_path=hsi_hdr_path,
            src_hsi_data_path=hsi_img_path,
            ndvi_dst_npy_path=ndvi_path,
            wavelengths=wavelengths,
            num_threads=NUM_THREADS,
            max_bytes=max(int(MEMORY * NDVI_MEM_FACTOR * 1e9), MIN_BYTES),
            zero_threshold=1e-9,
            show_progress=True,
        )
        print("NDVI calculation complete.")
        print(f"The NDVI result was saved to the ouput folder as:\n{ndvi_path.name}\n")

    ndvi = np.load(ndvi_path)
    if NDVI_BLUR_SIGMA > 0:
        ndvi = gaussian_filter(input=ndvi, sigma=NDVI_BLUR_SIGMA)

    ndvi_gray_img = Image.fromarray(
        np.clip(((ndvi[:, :, 0] + 1.0) / 2.0 * 255.0).round(), a_min=0, a_max=255).astype(np.uint8), mode="L"
    )
    ndvi_gray_img.save(IMG_OUTPUT_FOLDER / f"{HSI_NAME}_ndvi.png")
    del ndvi_gray_img

    where_ndvi_above_thresh = ndvi > NDVI_THRESHOLD

    where_ndvi_above_thresh_img = Image.fromarray(where_ndvi_above_thresh[:, :, 0].astype(np.uint8) * 255, mode="L")
    where_ndvi_above_thresh_img.save(IMG_OUTPUT_FOLDER / f"{HSI_NAME}_ndvi_abv_thrsh.png")
    del where_ndvi_above_thresh_img

    # ================================================================================================================
    # GEOMETRY
    # ----------------------------------------------------------------------------------------------------------------

    # Try to guess geo file path by shape

    def get_geo_path():
        vp_geo_shapes = {
            "vp_geo_hog_island_2019_0.npy": (3705, 3217, 3),
            "vp_geo_hog_island_2019_1.npy": (5321, 5417, 3),
        }

        geo_path = None
        for filename, shape in vp_geo_shapes.items():
            if (h, w) == shape[:2]:
                print(f"Shape matched to {filename},")
                geo_path = OUTPUT_FOLDER / filename
                break
        else:
            print("Shape does not match any known geometry file,")

        geo_path_message = "Please enter the path to a npy file containing the scene geometry information:\n"

        def prompt_geo_path():
            path = Path(input(geo_path_message))
            while not path.exists():
                path = Path(input(geo_path_message))
            return path

        if geo_path and geo_path.exists():
            if (
                input("Would you like to use this file for the solar and sensor geometry? (y/n) : ").strip().lower()
                != "y"
            ):
                geo_path = prompt_geo_path()
        elif geo_path:
            print(
                "This file was not found however. If you would like to use it, please either:"
                "\n1. Copy/download it to the 'output' folder. (easier)"
                f"\n   - from Google Drive : {GEO_GOOGLE_DRIVE_LINK}"
                f"\n   - from CIS Linux Systems : {GEO_CIS_LINUX_PATH}"
                "\n\n---OR---\n"
                "2. Generate it using the 'vp_mapping.ipynb' python notebook. (harder)\n"
            )
            if input("Would you like to continue manually? (y/n) : ").strip().lower() != "y":
                print("exiting...")
                sys.exit(0)
            geo_path = prompt_geo_path()
        else:
            if input("Would you like to enter the path manually? (y/n) : ").strip().lower() != "y":
                print("exiting...")
                sys.exit(0)
            geo_path = prompt_geo_path()
        return geo_path

    geo_path = get_geo_path()
    geo = np.load(geo_path)

    where_no_geo = np.expand_dims(np.any(np.isnan(geo), axis=2), axis=-1)

    geo_rgb_arr = geo.copy()
    geo_rgb_arr[:, :, 0][~where_no_geo[:, :, 0]] /= 90.0  # SZA
    geo_rgb_arr[:, :, 1][~where_no_geo[:, :, 0]] /= 90.0  # VZA
    geo_rgb_arr[:, :, 2][~where_no_geo[:, :, 0]] /= 360.0  # RAA
    geo_rgb_arr[~where_no_geo[:, :, 0]] = (
        geo_rgb_arr[~where_no_geo[:, :, 0]] * (1 - GEO_PNG_MISSING_PCT_DARKER) + GEO_PNG_MISSING_PCT_DARKER
    )
    geo_rgb_arr[where_no_geo[:, :, 0]] = 0
    geo_rgb_img = Image.fromarray((geo_rgb_arr * 255.0).round().astype(np.uint8), mode="RGB")
    geo_rgb_img.save(IMG_OUTPUT_FOLDER / f"{HSI_NAME}_geo.png")
    del geo_rgb_arr
    del geo_rgb_img

    # ================================================================================================================
    # MASK CREATION
    # ----------------------------------------------------------------------------------------------------------------

    inv_mask = ~where_dark & ~where_no_geo & where_ndvi_above_thresh

    inv_mask_img = Image.fromarray(inv_mask[:, :, 0].astype(np.uint8) * 255, mode="L")
    inv_mask_img.save(IMG_OUTPUT_FOLDER / f"{HSI_NAME}_inv_mask.png")
    del inv_mask_img

    inv_mask_mean = np.mean(inv_mask.astype(np.float64))
    if inv_mask_mean < MIN_TO_INVERT_PCT:
        raise RuntimeError(
            "The minimum percentage of pixels will not be inverted:"
            ""
            "inv_mask_mean must be less than MIN_TO_INVERT_PCT."
            ""
            f"MIN_TO_INVERT_PCT = {MIN_TO_INVERT_PCT}"
            f"    inv_mask_mean = {inv_mask_mean}"
            ""
            "Either increase MIN_TO_INVERT_PCT or try making the"
            "inv_mask more permissive by lowering the NDVI_THRESHOLD"
            "or increasing the NDVI_BLUR_SIGMA."
        )

    # ================================================================================================================
    # HSI GEO MASK STACK CREATION
    # ----------------------------------------------------------------------------------------------------------------

    hgms_path = OUTPUT_FOLDER / f"{HSI_NAME}_hgms.npy"
    if (
        not hgms_path.exists()
        or input(f"Would you like to use the last created hsi_geo_mask_stack for: {HSI_NAME} ? (y/n): ")
        .strip()
        .lower()
        != "y"
    ):
        print("\nCreating hsi_geo_mask_stack:")
        print("Copying the HSI to hsi_geo_mask_stack...")
        copy_add_channels_mp(
            src_hsi_hdr_path=hsi_hdr_path,
            src_hsi_data_path=hsi_img_path,
            dst_npy_path=hgms_path,
            num_channels=4,
            fill_value=None,
            add_to_back=True,
            num_threads=NUM_THREADS,
            max_bytes=max(int(MEMORY * HGMS_MEM_FACTOR * 1e9), MIN_BYTES),
            show_progress=True,
        )

    print("Copying the solar and sensor geometry along with the inversion mask to the hsi_geo_mask_stack...")
    with Memmap(npy_path=hgms_path, shape=None, dtype=dt, mode="r+") as hgms:
        if hgms.array is None:
            raise RuntimeError("hgms.array is None")
        hgms.array[:, :, -4:-1] = geo.astype(dt)
        hgms.array[:, :, -1] = inv_mask[:, :, 0].astype(dt)
    print("hsi_geo_mask_stack creation complete.")
    print(f"The hsi_geo_mask_stack result was saved to the ouput folder as:\n{hgms_path.name}\n")

    # ================================================================================================================
    # PROSAIL
    # ----------------------------------------------------------------------------------------------------------------

    print("\nInverting PROSAIL:")
    inv_path = OUTPUT_FOLDER / f"{HSI_NAME}_inv_res.npy"
    invert_prosail_mp(
        geo_mask_stack_src_npy_path=hgms_path,
        src_dtype=dt,
        inv_res_dst_npy_path=inv_path,
        wavelengths=wavelengths,
        min_wavelength=MIN_WAVELENGTH,
        max_wavelength=MAX_WAVELENGTH,
        atol_rmse_residual=ATOL_RMSE_RESIDUAL,
        atol_wavelength=ATOL_WAVELENGTH,
        maxiter_factor=MAXITER_FACTOR,
        is_adaptive=IS_ADAPTIVE,
        print_errors=PRINT_ERRORS,
        num_threads=NUM_THREADS,
        max_bytes=max(int(MEMORY * PROSAIL_MEM_FACTOR * 1e9), MIN_BYTES),
        show_progress=True,
    )
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
