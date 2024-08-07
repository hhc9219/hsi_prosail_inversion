from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from .typedefs import NDArrayFloat


def invert_prosail(
    hsi_geo_mask_stack: "NDArrayFloat",
    wavelengths: "NDArrayFloat",
    min_wavelength: float,
    max_wavelength: float,
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

    min_wavelength : float
        The minimum wavelength to consider for the inversion.

    max_wavelength : float
        The maximum wavelength to consider for the inversion.

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

    RuntimeError
        If the inversion fails for a pixel.
    """
    from numpy import array as np_array, zeros as np_zeros, float64 as np_float64
    from .prosail_data import ProsailData

    if hsi_geo_mask_stack.ndim != 2:
        raise NotImplementedError("invert_prosail currently only handles a single row of pixels (2D Array).")
    num_pixels, num_channels = hsi_geo_mask_stack.shape
    num_hsi_channels = len(wavelengths)
    if num_channels != num_hsi_channels + 4:
        raise RuntimeError("The provided hsi_geo_mask_stack has the incorrect number of channels.")

    for i in range(num_hsi_channels):
        val = wavelengths[i]
        if val >= min_wavelength:
            start_idx = i
            break

    for i in range(num_hsi_channels - 1, -1, -1):
        val = wavelengths[i]
        if val <= max_wavelength:
            end_idx = i + 1
            break

    wavs = wavelengths[start_idx:end_idx]
    hsi = hsi_geo_mask_stack[:, start_idx:end_idx]

    # geo has three channels 0 is solar_zenith, 1 is sensor_zenith, 2 is relative_azimuth
    geo = hsi_geo_mask_stack[:, num_hsi_channels:-1]
    float_mask = hsi_geo_mask_stack[:, -1]

    inversion_result = np_zeros(shape=(num_pixels, 9), dtype=np_float64)
    inversion_result[:, -1] = float_mask
    mask = float_mask.round().astype(bool)

    pd = ProsailData()
    initial_values = pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL

    for i in range(num_pixels):
        if mask[i]:
            try:
                success = pd.fit_to_reflectances(
                    wavelengths=wavs,
                    reflectances=hsi[i],
                    SZA=geo[i, 0],
                    VZA=geo[i, 1],
                    RAA=geo[i, 2],
                    atol_rmse_residual=atol_rmse_residual,
                    atol_wavelength=atol_wavelength,
                    maxiter_factor=maxiter_factor,
                    is_adaptive=is_adaptive,
                )
                inversion_result[i, :-1] = np_array(
                    [float(success), pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL], dtype=np_float64
                )
                if not success:
                    raise RuntimeError("PROSAIL inversion did not succeed.")
            except Exception as e:
                if print_errors:
                    print(f"Pixel {i} did not invert successfully. {e}")
            finally:
                pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL = initial_values
                pd.execute()
    return inversion_result


def invert_prosail_mp(
    geo_mask_stack_src_npy_path: "Path",
    src_dtype: type,
    inv_res_dst_npy_path: "Path",
    wavelengths: "NDArrayFloat",
    min_wavelength: float,
    max_wavelength: float,
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

    See for More Info
    --------
    :func:`invert_prosail`
    """
    from numpy import float64
    from .hsi_processing import make_hsi_func_npy_to_npy_mp

    invert_prosail_mp_func = make_hsi_func_npy_to_npy_mp(hsi_func=invert_prosail)
    invert_prosail_mp_func(
        src_npy_path=geo_mask_stack_src_npy_path,
        src_dtype=src_dtype,
        dst_npy_path=inv_res_dst_npy_path,
        dst_num_channels=9,
        dst_dtype=float64,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        atol_rmse_residual=atol_rmse_residual,
        atol_wavelength=atol_wavelength,
        maxiter_factor=maxiter_factor,
        is_adaptive=is_adaptive,
        print_errors=print_errors,
    )
