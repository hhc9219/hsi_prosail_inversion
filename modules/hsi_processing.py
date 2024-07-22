import colour
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue
from typing import Callable, Any

NDArrayFloat = np.ndarray[Any, np.dtype[np.float32 | np.float64]]


class ColorConverter:
    def __init__(
        self,
        original_wavelengths: NDArrayFloat,
        wavelengths_resample_interval: int | None = 1,
        illuminant=colour.SDS_ILLUMINANTS["D65"],
        cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    ):
        self.original_wavelengths = original_wavelengths

        self.wavelengths = (
            np.arange(
                round(original_wavelengths[0]),
                round(original_wavelengths[-1]) + wavelengths_resample_interval,
                wavelengths_resample_interval,
                dtype=np.int64,
            )
            if wavelengths_resample_interval is not None
            else original_wavelengths.astype(np.int64)
        )
        self.do_resample_reflectances = True if wavelengths_resample_interval is not None else False

        self.illuminant = illuminant
        self.cmfs = cmfs

    def resample_reflectances(self, reflectances: NDArrayFloat):
        return np.interp(self.wavelengths, self.original_wavelengths, reflectances)

    @staticmethod
    def xyz_to_sRGB(xyz: NDArrayFloat):

        return colour.XYZ_to_sRGB(xyz / 100).astype(xyz.dtype)

    def reflectances_to_xyz(
        self,
        reflectances: NDArrayFloat,
    ):
        refl = self.resample_reflectances(reflectances) if self.do_resample_reflectances else reflectances
        sd = colour.SpectralDistribution(data=refl, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return xyz

    def reflectances_to_sRGB_float(
        self,
        reflectances: NDArrayFloat,
    ) -> NDArrayFloat:
        xyz = self.reflectances_to_xyz(reflectances=reflectances)
        return self.xyz_to_sRGB(xyz)  # type: ignore

    def reflectances_to_sRGB(self, reflectances: NDArrayFloat):
        sRGB = self.reflectances_to_sRGB_float(reflectances=reflectances)
        return self.float_array_to_dc(sRGB)

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
        sRGB = np.apply_along_axis(func1d=self.reflectances_to_sRGB_float, axis=ax, arr=hsi)
        return sRGB

    def hsi_to_sRGB(self, hsi: NDArrayFloat):
        sRGB = self.hsi_to_sRGB_float(hsi)
        return self.float_array_to_dc(sRGB)


def hsi_to_sRGB_float(
    hsi: NDArrayFloat,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
    return color_converter.hsi_to_sRGB_float(hsi=hsi)


def hsi_to_sRGB(
    hsi: NDArrayFloat,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
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
    output_queue.cancel_join_thread()


def make_hsi_func_inplace_mp(hsi_func: Callable[[np.ndarray, Any], np.ndarray]):

    def mp_hsi_func(hsi: np.ndarray, num_threads: int, max_bytes=int(1e9), *args, **kwargs):
        h, w, _ = hsi.shape
        n = h * w
        progress_bar = tqdm(total=n, desc="Processing Pixels")
        for chunk in split_hsi_by_max_bytes(hsi=hsi, max_bytes=max_bytes):
            sub_chunks = [sub_chunk.copy() for sub_chunk in split_hsi_by_num_parts(hsi=chunk, num_parts=num_threads)]
            processes: list[Process] = []
            output_queue = Queue()
            for sub_chunk in sub_chunks:
                p = Process(
                    target=_process_sub_chunk,
                    args=(hsi_func, sub_chunk, output_queue, *args),
                    kwargs=kwargs,
                )
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            output_results = []
            while not output_queue.empty():
                output_results.append(output_queue.get())

            if output_results:
                output_results = np.concatenate(output_results, axis=0)
                d = output_results.shape[1]
                chunk[:, :d] = output_results

            n_chunk, _ = chunk.shape
            progress_bar.update(n_chunk)

        progress_bar.close()

    return mp_hsi_func


def hsi_to_sRGB_inplace_mp(
    hsi: np.ndarray,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
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
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
