import numpy as np
from .prosail_data import ProsailData
from .img_processing import make_img_func_mp
from .typedefs import NDArrayFloat


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
