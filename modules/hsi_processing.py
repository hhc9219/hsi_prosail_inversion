import colour
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue
from typing import Callable, Any

NDArrayFloat = np.ndarray[Any, np.dtype[np.float32 | np.float64]]


def float_img_to_dc(src: NDArrayFloat):
    return (src * 255).round().astype(np.uint8)


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
        return float_img_to_dc(sRGB)

    def hsi_to_sRGB_float(self, hsi: NDArrayFloat):
        if hsi.ndim == 3:
            ax = 2
        elif hsi.ndim == 2:
            ax = 1
        else:
            ValueError("src array must be either 2D or 3D.")
        sRGB = np.apply_along_axis(func1d=self.reflectances_to_sRGB_float, axis=ax, arr=hsi)
        return sRGB

    def hsi_to_sRGB(self, hsi: NDArrayFloat):
        sRGB = self.hsi_to_sRGB_float(hsi)
        return float_img_to_dc(sRGB)


def get_img_characteristics(src: np.ndarray):
    nd = src.ndim
    if nd == 3:
        h, w, d = src.shape
        n = h * w
    elif nd == 2:
        h, d = src.shape
        n = h
        w = 0
    elif nd == 1:
        d = src.shape[0]
        n = 1
        h = 0
        w = 0
    else:
        raise ValueError("src ndim is not 1, 2, or 3")
    dt = src.dtype
    s = src.size
    return h, w, d, dt, nd, s, n


def downsample_img(src: np.ndarray, new_height: int, new_width: int):
    h, w, d, dt, nd, s, n = get_img_characteristics(src=src)
    assert nd == 3
    row_step = h // new_height
    col_step = w // new_width
    resized_image = src[::row_step, ::col_step, :]
    resized_image = resized_image[:new_height, :new_width, :]
    return resized_image


def split_img_by_num_parts(src: np.ndarray, num_parts: int):
    h, w, d, dt, nd, s, n = get_img_characteristics(src=src)
    if nd == 3:
        flattened_view = src.reshape(-1, d)
    elif nd == 2:
        flattened_view = src
    else:
        ValueError("src array must be either 2D or 3D.")
    num_per_part = (n + num_parts - 1) // num_parts
    start = 0
    for i in range(num_parts - 1):
        end = start + num_per_part
        yield flattened_view[start:end]
        start = end
    yield flattened_view[start:]


def get_num_parts_from_max_bytes(src: np.ndarray, max_bytes: int):
    h, w, d, dt, nd, s, n = get_img_characteristics(src=src)
    bytes_per_pixel = d * src.itemsize
    max_pixels_per_part = max_bytes // bytes_per_pixel
    num_parts = n // max_pixels_per_part
    return num_parts


def split_img_by_max_bytes(src: np.ndarray, max_bytes: int):
    num_parts = get_num_parts_from_max_bytes(src=src, max_bytes=max_bytes)
    yield from split_img_by_num_parts(src=src, num_parts=num_parts)


def _process_sub_chunk(
    img_func: Callable[[np.ndarray, Any], np.ndarray],
    sub_chunk: np.ndarray,
    sub_chunk_num: int,
    output_queue: Queue,
    *args,
    **kwargs
):
    output_queue.put([sub_chunk_num, img_func(sub_chunk, *args, **kwargs)])


def make_img_func_mp(img_func: Callable[[np.ndarray, Any], np.ndarray]):

    def mp_img_func(
        src: np.ndarray, dst: np.ndarray, num_threads: int, max_bytes, show_progress: bool = True, *args, **kwargs
    ):
        if isinstance(dst, np.memmap):
            dst.flush()

        h_src, w_src, d_src, dt_src, nd_src, s_src, n_src = get_img_characteristics(src=src)
        h_dst, w_dst, d_dst, dt_dst, nd_dst, s_dst, n_dst = get_img_characteristics(src=dst)
        bpp_src, bpp_dst = d_src * src.itemsize, d_dst * dst.itemsize
        assert nd_src == 3
        assert nd_dst == 3
        assert h_src == h_dst
        assert w_src == w_dst

        if show_progress:
            progress_bar = tqdm(total=n_src, desc="Processing Pixels")
        output_queue = Queue()
        num_parts = get_num_parts_from_max_bytes(src=src if bpp_src > bpp_dst else dst, max_bytes=max_bytes)

        for chunk_src, chunk_dst in zip(
            split_img_by_num_parts(src=src, num_parts=num_parts), split_img_by_num_parts(src=dst, num_parts=num_parts)
        ):
            sub_chunks = [
                sub_chunk.copy() for sub_chunk in split_img_by_num_parts(src=chunk_src, num_parts=num_threads)
            ]

            sub_chunk_num = 0
            for sub_chunk in sub_chunks:
                p = Process(
                    target=_process_sub_chunk,
                    args=(img_func, sub_chunk, sub_chunk_num, output_queue, *args),
                    kwargs=kwargs,
                )
                p.start()
                sub_chunk_num += 1
            del sub_chunks

            output_results = [None] * num_threads
            for _ in range(num_threads):
                result = output_queue.get()
                output_results[result[0]] = result[1]

            if not output_queue.empty():
                raise RuntimeError("Multiprocess output queue was not completely emptied.")

            chunk_dst[:] = np.concatenate(output_results, axis=0, dtype=dt_dst)  # type: ignore
            del output_results
            if isinstance(chunk_dst, np.memmap):
                chunk_dst.flush()

            if show_progress:
                chunk_n, _ = chunk_src.shape
                progress_bar.update(chunk_n)

        output_queue.close()
        output_queue.join_thread()

        if show_progress:
            progress_bar.close()

    return mp_img_func


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


def hsi_to_sRGB_mp(
    hsi_src: np.ndarray,
    rgb_dst: np.ndarray,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    num_threads: int = 2,
    max_bytes=int(0.5e9),
    show_progress=True,
):
    hsi_to_sRGB_func = make_img_func_mp(hsi_to_sRGB)
    hsi_to_sRGB_func(
        src=hsi_src,
        dst=rgb_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
