import numpy as np
from .prosail_data import ProsailData
from .img_processing import make_img_func_mp
from .typedefs import NDArrayFloat


class ProsailInversionError(Exception):
    pass


def invert_prosail(
    hsi: NDArrayFloat,
    wavelengths: NDArrayFloat,
    atol_rmse_residual: float,
    atol_wavelength: float,
    maxiter_factor: int,
    black_threshold: float,
    ndvi_threshold: float,
    is_adaptive: bool,
    print_errors: bool,
):
    if hsi.ndim != 2:
        raise NotImplementedError("invert_prosail currently only handles a single row of pixels (2D Array).")
    num_pixels, num_channels = hsi.shape
    if num_channels != len(wavelengths):
        raise RuntimeError(
            "The provided hsi has the incorrect number of channels to match the number of provided wavelengths."
        )
    where_r = (wavelengths > 400) & (wavelengths < 700)
    where_nir = (wavelengths > 700) & (wavelengths < 1100)
    inversion_result = np.zeros(shape=(num_pixels, 11), dtype=np.float64)
    pd = ProsailData()
    initial_values = pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL, pd.SZA, pd.VZA, pd.RAA
    for i in range(num_pixels):
        try:
            if np.mean(hsi[i]) > black_threshold:
                r = np.mean(hsi[i][where_r])
                nir = np.mean(hsi[i][where_nir])
                ndvi = (nir - r) / (nir + r) if not abs(nir + r) < 1e-9 else 0
                if ndvi > ndvi_threshold:
                    success = pd.fit_to_reflectances(
                        wavelengths=wavelengths,
                        reflectances=hsi[i],
                        atol_rmse_residual=atol_rmse_residual,
                        atol_wavelength=atol_wavelength,
                        maxiter_factor=maxiter_factor,
                        is_adaptive=is_adaptive,
                    )
                    inversion_result[i] = np.array(
                        [
                            float(round(success)),
                            pd.N,
                            pd.CAB,
                            pd.CCX,
                            pd.EWT,
                            pd.LMA,
                            pd.LAI,
                            pd.PSOIL,
                            pd.SZA,
                            pd.VZA,
                            pd.RAA,
                        ],
                        dtype=np.float64,
                    )
                    if not success:
                        raise ProsailInversionError("PROSAIL inversion did not succeed.")
                else:
                    inversion_result[i, 0] = (
                        0.75  # for skipped pixels with ndvi less than ndvi_threshold success = 0.75
                    )
            else:
                inversion_result[i, 0] = 0.5  # for skipped black pixels success = 0.5
        except Exception as e:
            if print_errors:
                print(f"Pixel {i} did not invert successfully. {e}")
        finally:
            pd.N, pd.CAB, pd.CCX, pd.EWT, pd.LMA, pd.LAI, pd.PSOIL, pd.SZA, pd.VZA, pd.RAA = initial_values
            pd.execute()
    return inversion_result


def invert_prosail_mp(
    hsi_src: NDArrayFloat,
    inversion_result_dst: NDArrayFloat,
    wavelengths: NDArrayFloat,
    atol_rmse_residual: float,
    atol_wavelength: float,
    maxiter_factor: int,
    black_threshold: float,
    ndvi_threshold: float,
    is_adaptive: bool,
    num_threads: int,
    max_bytes: int,
    show_progress: bool = True,
    print_errors: bool = False,
):
    invert_prosail_mp_func = make_img_func_mp(img_func=invert_prosail)
    invert_prosail_mp_func(
        src=hsi_src,
        dst=inversion_result_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        atol_rmse_residual=atol_rmse_residual,
        atol_wavelength=atol_wavelength,
        maxiter_factor=maxiter_factor,
        black_threshold=black_threshold,
        ndvi_threshold=ndvi_threshold,
        is_adaptive=is_adaptive,
        print_errors=print_errors,
    )
