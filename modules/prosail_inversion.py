from typing import Any
import numpy as np
from .prosail_data import ProsailData
from .img_processing import make_img_func_mp
from .typedefs import NDArrayFloat, NDArrayBool


class ProsailInversionError(Exception):
    pass


def invert_prosail(
    hsi_geo_mask_stack: NDArrayFloat,
    wavelengths: NDArrayFloat,
    atol_rmse_residual: float,
    atol_wavelength: float,
    maxiter_factor: int,
    is_adaptive: bool,
    print_errors: bool,
):
    """
    Inverts the PROSAIL model for a single row of pixels from a given hyperspectral image (HSI) data stack.

    Parameters:
    ----------
    hsi_geo_mask_stack : NDArrayFloat
        A 2D numpy array with dimensions (num_pixels, num_channels) where:
        - num_pixels is the number of pixels in a single row.
        - num_channels is the total number of channels which must be equal to the number of wavelengths + 4.

        The structure of hsi_geo_mask_stack is as follows:
        - The first 'num_hsi_channels' contain the hyperspectral reflectance data.
        - The next three channels contain the geometric data:
          - Solar zenith angle (SZA) [0 to 90]
          - Sensor zenith angle (VZA) [0 to 90]
          - Relative azimuth angle (RAA) [0 to 360]
        - The last channel contains the mask data, which is a floating-point mask to indicate
          valid (1.0) or invalid (0.0) pixels for inversion.

    wavelengths : NDArrayFloat
        A 1D numpy array containing the wavelengths corresponding to the hyperspectral reflectance data.

    atol_rmse_residual : float
        The largest acceptable RMSE residual for fitting the reflectances.

    atol_wavelength : float
        The largest acceptable deviation in the wavelength for reflectance fitting.

    maxiter_factor : int
        The maximum number of iterations allowed for the nelder-mead simplex inversion is maxiter_factor
        times the number of inversion parameters.

    is_adaptive : bool
        If True, uses the adaptive nelder-mean implementation. Useful for inverting many parameters.

    print_errors : bool
        If True, prints errors for pixels that fail to invert successfully.

    Returns:
    -------
    inversion_result : NDArrayFloat
        A 2D numpy array with dimensions (num_pixels, 9) where:
        - The first column (index 0) contains a binary inversion success indicator (1.0 for success, 0.0 for failure)
        - The last column (index 8) contains the original floating-point mask values.
        - The columns in between (indices 1 to 7) contain the inversion results, which are:
          - Inverted PROSAIL parameters:
            - N:     (index 1)
            - CAB:   (index 2)
            - CCX:   (index 3)
            - EWT:   (index 4)
            - LMA:   (index 5)
            - LAI:   (index 6)
            - PSOIL: (index 7)

    Raises:
    ------
    NotImplementedError
        If hsi_geo_mask_stack is not a 2D array.

    RuntimeError
        If the number of channels in hsi_geo_mask_stack is incorrect.

    ProsailInversionError
        If the inversion fails for a pixel.
    """

    if hsi_geo_mask_stack.ndim != 2:
        raise NotImplementedError("invert_prosail currently only handles a single row of pixels (2D Array).")
    num_pixels, num_channels = hsi_geo_mask_stack.shape
    num_hsi_channels = len(wavelengths)
    if num_channels != num_hsi_channels + 4:
        raise RuntimeError("The provided hsi_geo_mask_stack has the incorrect number of channels.")
    hsi = hsi_geo_mask_stack[:, :num_hsi_channels]
    geo = hsi_geo_mask_stack[
        :, num_hsi_channels:-1
    ]  # geo has three channels 0 is solar_zenith, 1 is sensor_zenith, 2 is relative_azimuth
    float_mask = hsi_geo_mask_stack[:, -1]
    mask = float_mask.round().astype(bool)
    inversion_result = np.zeros(shape=(num_pixels, 9), dtype=np.float64)
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
                    RAA=geo[i, 2],
                    atol_rmse_residual=atol_rmse_residual,
                    atol_wavelength=atol_wavelength,
                    maxiter_factor=maxiter_factor,
                    is_adaptive=is_adaptive,
                )
                inversion_result[i, :-1] = np.array(
                    [float(success), pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL], dtype=np.float64
                )
                if not success:
                    raise ProsailInversionError("PROSAIL inversion did not succeed.")
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
    atol_rmse_residual: float,
    atol_wavelength: float,
    maxiter_factor: int,
    is_adaptive: bool,
    print_errors: bool,
    num_threads: int,
    max_bytes: int,
    show_progress: bool,
):
    """
    Parallelizes the inversion of the PROSAIL model for a given hyperspectral image (HSI) data stack.

    Parameters:
    ----------
    hsi_geo_mask_stack_src : NDArrayFloat
        A 3D numpy array with dimensions (h, w, d) where:
        - h is the height of the image.
        - w is the width of the image.
        - d is the total number of channels which must be equal to the number of wavelengths + 4.

        The structure of hsi_geo_mask_stack is as follows:
        - The first 'num_hsi_channels' contain the hyperspectral reflectance data.
        - The next three channels contain the geometric data:
          - Solar zenith angle (SZA) [0 to 90]
          - Sensor zenith angle (VZA) [0 to 90]
          - Relative azimuth angle (RAA) [0 to 360]
        - The last channel contains the mask data, which is a floating-point mask to indicate
          valid (1.0) or invalid (0.0) pixels for inversion.

    inversion_result_dst : NDArrayFloat
        A 3D numpy array with dimensions (h, w, 9) where:
        - The first channel (index 0) contains a binary inversion success indicator (1.0 for success, 0.0 for failure)
        - The last channel (index 8) contains the original floating-point mask values.
        - The channels in between (indices 1 to 7) contain the inversion results, which are:
          - Inverted PROSAIL parameters:
            - N:     (index 1)
            - CAB:   (index 2)
            - CCX:   (index 3)
            - EWT:   (index 4)
            - LMA:   (index 5)
            - LAI:   (index 6)
            - PSOIL: (index 7)

    wavelengths : NDArrayFloat
        A 1D numpy array containing the wavelengths corresponding to the hyperspectral reflectance data.

    atol_rmse_residual : float
        The largest acceptable RMSE residual for fitting the reflectances.

    atol_wavelength : float
        The largest acceptable deviation in the wavelength for reflectance fitting.

    maxiter_factor : int
        The maximum number of iterations allowed for the nelder-mead simplex inversion is maxiter_factor
        times the number of inversion parameters.

    is_adaptive : bool
        If True, uses the adaptive nelder-mean implementation. Useful for inverting many parameters.

    print_errors : bool, optional
        If True, prints errors for pixels that fail to invert successfully. Default is False.

    num_threads : int
        The number of threads to use for parallel processing.

    max_bytes : int
        The target number of bytes to be used while processing a sub-section of the HSI.

    show_progress : bool, optional
        If True, displays a progress bar. Default is True.

    Returns:
    -------
    None
    """
    invert_prosail_mp_func = make_img_func_mp(img_func=invert_prosail)
    invert_prosail_mp_func(
        src=hsi_geo_mask_stack_src,
        dst=inversion_result_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        atol_rmse_residual=atol_rmse_residual,
        atol_wavelength=atol_wavelength,
        maxiter_factor=maxiter_factor,
        is_adaptive=is_adaptive,
        print_errors=print_errors,
    )


def where_dark(hsi: NDArrayFloat, dark_threshold=1e-9):
    if hsi.ndim != 2:
        raise NotImplementedError("where_dark currently only handles a single row of pixels (2D Array).")
    dark_elements = hsi < dark_threshold
    return np.expand_dims(np.all(dark_elements, axis=1), axis=-1)


def where_dark_mp(
    hsi: NDArrayFloat,
    where_dark_dst: NDArrayBool,
    num_threads: int,
    max_bytes: int,
    dark_threshold=1e-9,
    show_progress=True,
):
    where_dark_mp_func = make_img_func_mp(img_func=where_dark)
    where_dark_mp_func(
        src=hsi,
        dst=where_dark_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        dark_threshold=dark_threshold,
    )


def calculate_ndvi(hsi: NDArrayFloat, wavelengths: NDArrayFloat, zero_threshold=1e-9):
    if hsi.ndim != 2:
        raise NotImplementedError("calculate_ndvi currently only handles a single row of pixels (2D Array).")
    num_pixels, num_channels = hsi.shape
    if num_channels != len(wavelengths):
        raise RuntimeError(
            "The provided hsi has the incorrect number of channels to match the number of provided wavelengths."
        )
    where_r = (wavelengths > 400) & (wavelengths < 700)
    where_nir = (wavelengths > 700) & (wavelengths < 1100)
    r = np.mean(hsi[:, where_r], axis=1)
    nir = np.mean(hsi[:, where_nir], axis=1)
    ndvi = np.zeros(shape=(num_pixels, 1), dtype=np.float64)
    where_not_r_nir_0 = ~(np.abs(nir + r) < zero_threshold)
    ndvi[where_not_r_nir_0, 0] = (nir[where_not_r_nir_0] - r[where_not_r_nir_0]) / (
        nir[where_not_r_nir_0] + r[where_not_r_nir_0]
    )
    return ndvi


def calculate_ndvi_mp(
    hsi: NDArrayFloat,
    ndvi_dst: np.ndarray[Any, np.dtype[np.float64]],
    wavelengths: NDArrayFloat,
    num_threads: int,
    max_bytes: int,
    zero_threshold=1e-9,
    show_progress: bool = True,
):
    calculate_ndvi_mp_func = make_img_func_mp(img_func=calculate_ndvi)
    calculate_ndvi_mp_func(
        src=hsi,
        dst=ndvi_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        zero_threshold=zero_threshold,
    )
