"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""

from modules.environment_manager import get_persistent_config_data

THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)

# Imports
import numpy as np
from pathlib import Path
from typing import Any
from modules import hsi_io
from modules.prosail_data import ProsailData
from modules.hsi_processing import make_img_func_mp

NDArrayFloat = np.ndarray[Any, np.dtype[np.float32 | np.float64]]


def main():

    img_name = input("Please enter the hsi to process from hsi_config.json: ")

    hsi_hdr_path = Path(HSI_CONFIG[img_name]["hdr"])
    hsi_img_path = Path(HSI_CONFIG[img_name]["img"])

    wavelengths = hsi_io.get_wavelengths(hsi_hdr_path)

    hsi = hsi_io.open_envi_hsi_as_np_memmap(img_hdr_path=hsi_hdr_path, img_data_path=hsi_img_path, writable=False)
    h, w, _ = hsi.shape
    with hsi_io.Memmap(
        OUTPUT_FOLDER / "inversion_result.npy", shape=(h, w, 9), dtype=np.float64, mode="w+"
    ) as inversion_result:
        if inversion_result.array is not None:

            invert_prosail_mp(
                hsi_geo_mask_stack_src=hsi,
                inversion_result_dst=inversion_result.array,
                wavelengths=wavelengths,
                num_threads=2,
                max_bytes=int(0.3e9),
            )

    del hsi


def invert_prosail(hsi_geo_mask_stack: NDArrayFloat, wavelengths: NDArrayFloat, print_errors: bool):
    if hsi_geo_mask_stack.ndim == 3:
        raise NotImplementedError("invert_prosail currently only handles a single row of pixels.")
    num_pixels = hsi_geo_mask_stack.shape[0]
    num_hsi_channels = len(wavelengths)
    hsi = hsi_geo_mask_stack[:, :num_hsi_channels]
    geo = hsi_geo_mask_stack[
        :, num_hsi_channels:-1
    ]  # geo has three channels 0 is solar_zenith, 1 is sensor_zenith, 2 is sensor_azimuth
    float_mask = hsi_geo_mask_stack[:, -1]
    mask = float_mask.round().astype(bool)
    inversion_result = np.empty(shape=(num_pixels, 9), dtype=np.float64)
    inversion_result[:, 0] = (~mask).astype(np.float64)
    inversion_result[:, -1] = float_mask
    pd = ProsailData()
    initial_values = pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL
    for i in range(num_pixels):
        if mask[i]:
            try:
                success = pd.fit_to_reflectances(
                    wavelengths=wavelengths,
                    reflectances=hsi[i],
                    SZA=geo[i, 0],
                    VZA=geo[i, 1],
                    RAA=(geo[i, 2] - geo[i, 0]),
                )
                inversion_result[i, :-1] = np.array(
                    [float(success), pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL], dtype=np.float64
                )
                if print_errors and not success:
                    print(f"Pixel {i} did not invert successfully.")
            except Exception as e:
                if print_errors:
                    print(f"Pixel {i} did not invert successfully. {e}")
            finally:
                pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL = initial_values
                pd.execute()
    return inversion_result


def invert_prosail_mp(
    hsi_geo_mask_stack_src: NDArrayFloat,
    inversion_result_dst: NDArrayFloat,
    wavelengths: NDArrayFloat,
    num_threads: int,
    max_bytes: int,
    show_progress: bool = True,
    print_errors: bool = False,
):
    invert_prosail_mp_func = make_img_func_mp(img_func=invert_prosail)
    invert_prosail_mp_func(
        src=hsi_geo_mask_stack_src,
        dst=inversion_result_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        print_errors=print_errors,
    )


if __name__ == "__main__":
    main()
