import colour
import numpy as np
from multiprocessing import Process, Queue
from typing import Callable, Any

NDArrayFloat = np.ndarray[Any, np.dtype[np.float32 | np.float64]]


class ColorConverter:
    def __init__(
        self,
        wavelengths: NDArrayFloat,
        illuminant=colour.SDS_ILLUMINANTS["D65"],
        cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    ):
        self.wavelengths = wavelengths
        self.illuminant = illuminant
        self.cmfs = cmfs

    @staticmethod
    def xyz_to_sRGB(xyz: NDArrayFloat):

        return colour.XYZ_to_sRGB(xyz / 100).astype(xyz.dtype)

    def reflectances_to_xyz(
        self,
        reflectances: NDArrayFloat,
    ):
        sd = colour.SpectralDistribution(data=reflectances, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return xyz

    def reflectances_to_sRGB(
        self,
        reflectances: NDArrayFloat,
    ) -> NDArrayFloat:
        sd = colour.SpectralDistribution(data=reflectances, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return self.xyz_to_sRGB(xyz)  # type: ignore

    @staticmethod
    def float_array_to_dc(float_array: NDArrayFloat):
        return (float_array * 255).round().astype(np.uint8)

    def hsi_to_sRGB_float(self, hsi: NDArrayFloat):
        if hsi.ndim == 3:
            ax = 2
        elif hsi.ndim == 2:
            ax = 1
        else:
            ValueError("HSI array must be either 2D or 3D.")
        sRGB = np.apply_along_axis(func1d=self.reflectances_to_sRGB, axis=ax, arr=hsi)
        return sRGB

    def hsi_to_sRGB(self, hsi: NDArrayFloat):
        sRGB = self.hsi_to_sRGB_float(hsi)
        return self.float_array_to_dc(sRGB)


def hsi_to_sRGB_float(
    hsi: NDArrayFloat,
    wavelengths: NDArrayFloat,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(wavelengths=wavelengths, illuminant=illuminant, cmfs=cmfs)
    return color_converter.hsi_to_sRGB_float(hsi=hsi)


def hsi_to_sRGB(
    hsi: NDArrayFloat,
    wavelengths: NDArrayFloat,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(wavelengths=wavelengths, illuminant=illuminant, cmfs=cmfs)
    return color_converter.hsi_to_sRGB(hsi=hsi)


def downsample(hsi: NDArrayFloat, new_height: int, new_width: int):
    original_height, original_width, channels = hsi.shape
    row_step = original_height // new_height
    col_step = original_width // new_width
    resized_image = hsi[::row_step, ::col_step, :]
    resized_image = resized_image[:new_height, :new_width, :]
    return resized_image


def split_hsi_by_num_parts(hsi: np.ndarray, num_parts: int):
    if hsi.ndim == 3:
        num_channels = hsi.shape[-1]
        flattened_view = hsi.reshape(-1, num_channels)
    elif hsi.ndim == 2:
        flattened_view = hsi
    else:
        raise ValueError("HSI array must be either 2D or 3D.")
    n = flattened_view.shape[0]
    num_per_part = n // num_parts
    start = 0
    for i in range(num_parts - 1):
        end = start + num_per_part
        yield flattened_view[start:end]
        start = end
    yield flattened_view[start:]


def split_hsi_by_max_bytes(hsi: np.ndarray, max_bytes: int):
    if hsi.ndim == 3:
        h, w, d = hsi.shape
        n = h * w
    elif hsi.ndim == 2:
        h, d = hsi.shape
        n = h
    else:
        raise ValueError("HSI array must be either 2D or 3D.")
    bytes_per_pixel = d * hsi.itemsize
    max_pixels_per_part = max_bytes // bytes_per_pixel
    num_parts = (n + max_pixels_per_part - 1) // max_pixels_per_part
    yield from split_hsi_by_num_parts(hsi=hsi, num_parts=num_parts)


def _process_sub_chunk(
    hsi_func: Callable[[np.ndarray, Any], np.ndarray], sub_chunk: np.ndarray, output_queue: Queue, *args, **kwargs
):
    output_queue.put(hsi_func(sub_chunk, *args, **kwargs))


def make_hsi_func_inplace_mp(hsi_func: Callable[[np.ndarray, Any], np.ndarray]):
    """
    Creates a multiprocessing wrapper for an HSI processing function.

    This function generates a new function that applies the given HSI processing
    function ('hsi_func') in parallel using multiple processes. The generated
    function modifies the input HSI array in place.

    Parameters:
    -----------
    hsi_func : Callable[[np.ndarray, Any], np.ndarray]
        A function that processes an HSI array. The function should accept the HSI
        array as its first argument and return the modified HSI array with the
        same height, width, and datatype.

    Returns:
    --------
    Callable[[np.ndarray, int, int, Any], None]
        A function that applies the given HSI processing function in parallel
        using multiple processes. The returned function has the following
        parameters:

        - hsi : np.ndarray
            The hyperspectral image array to be processed.
        - num_threads : int
            The number of parallel processes to use.
        - max_bytes : int
            The maximum size of data (in bytes) to be processed in each chunk.
            Defaults to 1e9 (1 GB).
        - *args : Any
            Additional positional arguments to pass to the HSI processing function.
        - **kwargs : Any
            Additional keyword arguments to pass to the HSI processing function.

    Notes:
    ------
    The function assumes that the input HSI array has three dimensions
    (height, width, channels).

    The input HSI array is split into smaller chunks to ensure that the size
    of each chunk does not exceed `max_bytes`. Each chunk is then further
    divided among the specified number of threads, and the processing is
    carried out in parallel.
    """

    def mp_hsi_func(hsi: np.ndarray, num_threads: int, max_bytes=int(1e9), *args, **kwargs):
        for chunk in split_hsi_by_max_bytes(hsi=hsi, max_bytes=max_bytes):
            processes: list[Process] = []
            output_queue = Queue(maxsize=num_threads)
            for sub_chunk in split_hsi_by_num_parts(hsi=chunk.copy(), num_parts=num_threads):
                p = Process(target=_process_sub_chunk, args=(hsi_func, sub_chunk, output_queue, *args), kwargs=kwargs)
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            output_results = []
            while not output_queue.empty():
                output_results.append(output_queue.get())
            output_results = np.concatenate(output_results, axis=0)
            d = output_results.shape[1]
            chunk[:, :d] = output_results

    return mp_hsi_func


def hsi_to_sRGB_inplace_mp(
    hsi: np.ndarray,
    wavelengths: NDArrayFloat,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    num_threads: int = 2,
    max_bytes=int(1e9),
):
    hsi_to_sRGB_func = make_hsi_func_inplace_mp(hsi_to_sRGB_float)
    hsi_to_sRGB_func(
        hsi=hsi,
        num_threads=num_threads,
        max_bytes=max_bytes,
        wavelengths=wavelengths,
        illuminant=illuminant,
        cmfs=cmfs,
    )
