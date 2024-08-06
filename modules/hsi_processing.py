from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from pathlib import Path
    import numpy as np
    from .typedefs import NDArrayFloat


def _process_sub_chunk(
    hsi_func: "Callable[..., np.ndarray]",
    sub_chunk_slice: slice,
    src_chunk_shm_name: str,
    dst_chunk_shm_name: str,
    src_chunk_shm_a_shape: tuple,
    src_dtype: "np.dtype",
    dst_chunk_shm_a_shape: tuple,
    dst_dtype: "np.dtype",
    *args,
    **kwargs,
):
    from multiprocessing import shared_memory
    from numpy import ndarray

    # Load
    existing_src_chunk_shm = shared_memory.SharedMemory(name=src_chunk_shm_name)
    existing_dst_chunk_shm = shared_memory.SharedMemory(name=dst_chunk_shm_name)
    existing_src_chunk_shm_a = ndarray(shape=src_chunk_shm_a_shape, dtype=src_dtype, buffer=existing_src_chunk_shm.buf)
    existing_dst_chunk_shm_a = ndarray(shape=dst_chunk_shm_a_shape, dtype=dst_dtype, buffer=existing_dst_chunk_shm.buf)

    try:
        # Call hsi_func
        existing_dst_chunk_shm_a[sub_chunk_slice] = hsi_func(
            existing_src_chunk_shm_a[sub_chunk_slice], *args, **kwargs
        )
    except Exception as e:
        print(f"An error occurred while processing a sub-section of the image: {e}")
    finally:
        # Cleanup
        del existing_src_chunk_shm_a
        del existing_dst_chunk_shm_a
        existing_src_chunk_shm.close()
        existing_dst_chunk_shm.close()


def make_hsi_func_envi_to_npy_mp(hsi_func: "Callable[..., np.ndarray]"):
    from multiprocessing import Process, shared_memory
    from numpy import ndarray
    from tqdm import tqdm
    from .npmemmap import Memmap
    from .hsi_io import open_envi_hsi_as_np_memmap
    from .img_processing import get_img_characteristics, get_num_parts_from_max_bytes, split_img_by_num_parts_2d_slices

    def mp_hsi_func(
        src_hsi_hdr_path: "Path",
        src_hsi_data_path: "Path",
        dst_npy_path: "Path",
        dst_num_channels: int,
        dst_dtype: type,
        num_threads: int,
        max_bytes: int,
        show_progress: bool = True,
        *args,
        **kwargs,
    ):

        # Open src to get characteristics
        src = open_envi_hsi_as_np_memmap(
            img_hdr_path=src_hsi_hdr_path, img_data_path=src_hsi_data_path, writable=False
        )
        try:
            h_src, w_src, d_src, dt_src, nd_src, s_src, n_src = get_img_characteristics(src=src)
            assert nd_src == 3  # Assert the src is 3D
            bpp_src = d_src * src.itemsize

            # Open dst to get characteristics
            assert dst_num_channels >= 1  # Assert the dst has at least one channel
            dst = Memmap(npy_path=dst_npy_path, shape=(h_src, w_src, dst_num_channels), dtype=dst_dtype, mode="a+")
            with dst:
                if dst.array is None:
                    raise RuntimeError("dst.array is None")
                h_dst, w_dst, d_dst, dt_dst, nd_dst, s_dst, n_dst = get_img_characteristics(src=dst.array)
                bpp_dst = d_dst * dst.array.itemsize

                # Use bytes per pixel of src and dst to determine how many chunks need to be loaded into memory independently
                num_parts = get_num_parts_from_max_bytes(
                    src=src if bpp_src > bpp_dst else dst.array, max_bytes=(max_bytes // 2)
                )

                # Create dst shared memory array
                chunk_dst_slices = list(
                    split_img_by_num_parts_2d_slices(src=dst.array.reshape(-1, d_dst), num_parts=num_parts)
                )
                first_chunk_dst = dst.array.reshape(-1, d_dst)[chunk_dst_slices[0]]
                try:
                    first_chunk_dst_shape = first_chunk_dst.shape
                    dst_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_dst.nbytes)
                    dst_chunk_shm_a = ndarray(first_chunk_dst_shape, dtype=dt_dst, buffer=dst_chunk_shm.buf)
                    dst_chunk_shm_name = dst_chunk_shm.name
                finally:
                    # Free memory
                    del first_chunk_dst

            # Create src shared memory array
            chunk_src_slices = list(split_img_by_num_parts_2d_slices(src=src.reshape(-1, d_src), num_parts=num_parts))
            first_chunk_src = src.reshape(-1, d_src)[chunk_src_slices[0]]
            try:
                first_chunk_src_shape = first_chunk_src.shape
                src_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_src.nbytes)
                src_chunk_shm_a = ndarray(first_chunk_src_shape, dtype=dt_src, buffer=src_chunk_shm.buf)
                src_chunk_shm_name = src_chunk_shm.name
            finally:
                # Free memory
                del first_chunk_src

        except Exception as e:
            raise RuntimeError(f"An error occured during multiprocessing setup: {e}")
        finally:
            # Free memory
            del src

        # Create a progress bar that updates with each processed chunk
        if show_progress:
            progress_bar = tqdm(total=n_src, desc="Processing Pixels")

        try:
            # Iteratively yield chunk slices of the src and dst
            for chunk_src_slice, chunk_dst_slice in zip(chunk_src_slices, chunk_dst_slices):

                # Copy chunk_src to shared memory src
                whole_src = open_envi_hsi_as_np_memmap(
                    img_hdr_path=src_hsi_hdr_path, img_data_path=src_hsi_data_path, writable=False
                )
                chunk_src = whole_src.reshape(-1, d_src)[chunk_src_slice]
                try:
                    chunk_n, _ = chunk_src.shape
                    src_chunk_shm_a[:chunk_n] = chunk_src[:chunk_n]
                    sub_chunk_slices = list(split_img_by_num_parts_2d_slices(src=chunk_src, num_parts=num_threads))
                finally:
                    # Free memory
                    del chunk_src
                    del whole_src

                # Spawn processes to compute the result for each sub chunk
                processes: list[Process] = []
                for sub_chunk_slice in sub_chunk_slices:
                    process = Process(
                        target=_process_sub_chunk,
                        args=(
                            hsi_func,
                            sub_chunk_slice,
                            src_chunk_shm_name,
                            dst_chunk_shm_name,
                            first_chunk_src_shape,
                            dt_src,
                            first_chunk_dst_shape,
                            dt_dst,
                            *args,
                        ),
                        kwargs=kwargs,
                    )
                    processes.append(process)
                    process.start()

                # Join and free all process resources
                for p in processes:
                    p.join()
                    p.close()

                # Copy shared memory dst to chunk_dst
                with dst:
                    dst.array.reshape(-1, d_dst)[chunk_dst_slice][:chunk_n] = dst_chunk_shm_a[:chunk_n]

                # Update the progress bar to reflect the number of pixels processed within the chunk
                if show_progress:
                    progress_bar.update(chunk_n)

        except Exception as e:
            print(f"An error occured while multiprocessing the image: {e}")

        finally:
            # Shared memory cleanup
            del src_chunk_shm_a
            del dst_chunk_shm_a
            src_chunk_shm.close()
            dst_chunk_shm.close()
            src_chunk_shm.unlink()
            dst_chunk_shm.unlink()

            # Cleanup and close the progress bar
            if show_progress:
                progress_bar.close()

    return mp_hsi_func


def make_hsi_func_npy_to_npy_mp(hsi_func: "Callable[..., np.ndarray]"):
    from multiprocessing import Process, shared_memory
    from numpy import ndarray
    from tqdm import tqdm
    from .npmemmap import Memmap
    from .img_processing import get_img_characteristics, get_num_parts_from_max_bytes, split_img_by_num_parts_2d_slices

    def mp_hsi_func(
        src_npy_path: "Path",
        src_dtype: type,
        dst_npy_path: "Path",
        dst_num_channels: int,
        dst_dtype: type,
        num_threads: int,
        max_bytes: int,
        show_progress: bool = True,
        *args,
        **kwargs,
    ):

        # Open src to get characteristics
        src = Memmap(npy_path=src_npy_path, shape=None, dtype=src_dtype, mode="r")
        with src:
            if src.array is None:
                raise RuntimeError("src.array is None")
            h_src, w_src, d_src, dt_src, nd_src, s_src, n_src = get_img_characteristics(src=src.array)
            assert nd_src == 3  # Assert the src is 3D
            bpp_src = d_src * src.array.itemsize

            # Open dst to get characteristics
            assert dst_num_channels >= 1  # Assert the dst has at least one channel
            dst = Memmap(npy_path=dst_npy_path, shape=(h_src, w_src, dst_num_channels), dtype=dst_dtype, mode="a+")
            with dst:
                if dst.array is None:
                    raise RuntimeError("dst.array is None")
                h_dst, w_dst, d_dst, dt_dst, nd_dst, s_dst, n_dst = get_img_characteristics(src=dst.array)
                bpp_dst = d_dst * dst.array.itemsize

                # Use bytes per pixel of src and dst to determine how many chunks need to be loaded into memory independently
                num_parts = get_num_parts_from_max_bytes(
                    src=src.array if bpp_src > bpp_dst else dst.array, max_bytes=(max_bytes // 2)
                )

                # Create dst shared memory array
                chunk_dst_slices = list(
                    split_img_by_num_parts_2d_slices(src=dst.array.reshape(-1, d_dst), num_parts=num_parts)
                )
                first_chunk_dst = dst.array.reshape(-1, d_dst)[chunk_dst_slices[0]]
                try:
                    first_chunk_dst_shape = first_chunk_dst.shape
                    dst_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_dst.nbytes)
                    dst_chunk_shm_a = ndarray(first_chunk_dst_shape, dtype=dt_dst, buffer=dst_chunk_shm.buf)
                    dst_chunk_shm_name = dst_chunk_shm.name
                finally:
                    # Free memory
                    del first_chunk_dst

            # Create src shared memory array
            chunk_src_slices = list(
                split_img_by_num_parts_2d_slices(src=src.array.reshape(-1, d_src), num_parts=num_parts)
            )
            first_chunk_src = src.array.reshape(-1, d_src)[chunk_src_slices[0]]
            try:
                first_chunk_src_shape = first_chunk_src.shape
                src_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_src.nbytes)
                src_chunk_shm_a = ndarray(first_chunk_src_shape, dtype=dt_src, buffer=src_chunk_shm.buf)
                src_chunk_shm_name = src_chunk_shm.name
            finally:
                # Free memory
                del first_chunk_src

        # Create a progress bar that updates with each processed chunk
        if show_progress:
            progress_bar = tqdm(total=n_src, desc="Processing Pixels")

        try:
            # Iteratively yield chunk slices of the src and dst
            for chunk_src_slice, chunk_dst_slice in zip(chunk_src_slices, chunk_dst_slices):

                # Copy chunk_src to shared memory src
                with src:
                    chunk_src = src.array.reshape(-1, d_src)[chunk_src_slice]
                    try:
                        chunk_n, _ = chunk_src.shape
                        src_chunk_shm_a[:chunk_n] = chunk_src[:chunk_n]
                        sub_chunk_slices = list(split_img_by_num_parts_2d_slices(src=chunk_src, num_parts=num_threads))
                    finally:
                        # Free memory
                        del chunk_src

                # Spawn processes to compute the result for each sub chunk
                processes: list[Process] = []
                for sub_chunk_slice in sub_chunk_slices:
                    process = Process(
                        target=_process_sub_chunk,
                        args=(
                            hsi_func,
                            sub_chunk_slice,
                            src_chunk_shm_name,
                            dst_chunk_shm_name,
                            first_chunk_src_shape,
                            dt_src,
                            first_chunk_dst_shape,
                            dt_dst,
                            *args,
                        ),
                        kwargs=kwargs,
                    )
                    processes.append(process)
                    process.start()

                # Join and free all process resources
                for p in processes:
                    p.join()
                    p.close()

                # Copy shared memory dst to chunk_dst
                with dst:
                    dst.array.reshape(-1, d_dst)[chunk_dst_slice][:chunk_n] = dst_chunk_shm_a[:chunk_n]

                # Update the progress bar to reflect the number of pixels processed within the chunk
                if show_progress:
                    progress_bar.update(chunk_n)

        except Exception as e:
            print(f"An error occured while multiprocessing the image: {e}")

        finally:
            # Shared memory cleanup
            del src_chunk_shm_a
            del dst_chunk_shm_a
            src_chunk_shm.close()
            dst_chunk_shm.close()
            src_chunk_shm.unlink()
            dst_chunk_shm.unlink()

            # Cleanup and close the progress bar
            if show_progress:
                progress_bar.close()

    return mp_hsi_func


def copy_add_channels(hsi: "NDArrayFloat", num_channels: int, fill_value: float | int | None, add_to_back: bool):
    from numpy import full as np_full, empty as np_empty, concatenate as np_concatenate

    if hsi.ndim != 2:
        raise NotImplementedError("copy_add_channels currently only handles a single row of pixels (2D Array).")
    new_channels = (
        np_full(shape=(hsi.shape[0], num_channels), dtype=hsi.dtype, fill_value=fill_value)
        if fill_value is not None
        else np_empty(shape=(hsi.shape[0], num_channels), dtype=hsi.dtype)
    )
    return (
        np_concatenate([hsi, new_channels], axis=1, dtype=hsi.dtype)
        if add_to_back
        else np_concatenate([new_channels, hsi], axis=1, dtype=hsi.dtype)
    )


def copy_add_channels_mp(
    src_hsi_hdr_path: "Path",
    src_hsi_data_path: "Path",
    dst_npy_path: "Path",
    num_channels: int,
    fill_value: float | int | None,
    add_to_back: bool,
    num_threads: int,
    max_bytes: int,
    show_progress: bool = True,
):
    from .hsi_io import open_envi_hsi_as_np_memmap

    src_hsi = open_envi_hsi_as_np_memmap(
        img_hdr_path=src_hsi_hdr_path, img_data_path=src_hsi_data_path, writable=False
    )
    try:
        assert src_hsi.ndim == 3
        src_hsi_channels = src_hsi.shape[2]
        src_hsi_dtype = src_hsi.dtype
    finally:
        del src_hsi
    copy_add_channels_mp_func = make_hsi_func_envi_to_npy_mp(hsi_func=copy_add_channels)
    copy_add_channels_mp_func(
        src_hsi_hdr_path=src_hsi_hdr_path,
        src_hsi_data_path=src_hsi_data_path,
        dst_npy_path=dst_npy_path,
        dst_num_channels=src_hsi_channels + num_channels,
        dst_dtype=src_hsi_dtype,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        num_channels=num_channels,
        fill_value=fill_value,
        add_to_back=add_to_back,
    )


def where_dark(hsi: "NDArrayFloat", dark_threshold=1e-9):
    from numpy import expand_dims as np_expand_dims, all as np_all

    if hsi.ndim != 2:
        raise NotImplementedError("where_dark currently only handles a single row of pixels (2D Array).")
    dark_elements = hsi < dark_threshold
    return np_expand_dims(np_all(dark_elements, axis=1), axis=-1)


def where_dark_mp(
    src_hsi_hdr_path: "Path",
    src_hsi_data_path: "Path",
    where_dark_dst_npy_path: "Path",
    num_threads: int,
    max_bytes: int,
    dark_threshold=1e-9,
    show_progress=True,
):
    where_dark_mp_func = make_hsi_func_envi_to_npy_mp(hsi_func=where_dark)
    where_dark_mp_func(
        src_hsi_hdr_path=src_hsi_hdr_path,
        src_hsi_data_path=src_hsi_data_path,
        dst_npy_path=where_dark_dst_npy_path,
        dst_num_channels=1,
        dst_dtype=bool,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        dark_threshold=dark_threshold,
    )


def calculate_ndvi(hsi: "NDArrayFloat", wavelengths: "NDArrayFloat", zero_threshold=1e-9):
    from numpy import mean as np_mean, zeros as np_zeros, abs as np_abs, float64 as np_float64

    if hsi.ndim != 2:
        raise NotImplementedError("calculate_ndvi currently only handles a single row of pixels (2D Array).")
    num_pixels, num_channels = hsi.shape
    if num_channels != len(wavelengths):
        raise RuntimeError(
            "The provided hsi has the incorrect number of channels to match the number of provided wavelengths."
        )
    where_r = (wavelengths > 400) & (wavelengths < 700)
    where_nir = (wavelengths > 700) & (wavelengths < 1100)
    r = np_mean(hsi[:, where_r], axis=1)
    nir = np_mean(hsi[:, where_nir], axis=1)
    ndvi = np_zeros(shape=(num_pixels, 1), dtype=np_float64)
    where_not_r_nir_0 = ~(np_abs(nir + r) < zero_threshold)
    ndvi[where_not_r_nir_0, 0] = (nir[where_not_r_nir_0] - r[where_not_r_nir_0]) / (
        nir[where_not_r_nir_0] + r[where_not_r_nir_0]
    )
    return ndvi


def calculate_ndvi_mp(
    src_hsi_hdr_path: "Path",
    src_hsi_data_path: "Path",
    ndvi_dst_npy_path: "Path",
    wavelengths: "NDArrayFloat",
    num_threads: int,
    max_bytes: int,
    zero_threshold=1e-9,
    show_progress: bool = True,
):
    from numpy import float64

    calculate_ndvi_mp_func = make_hsi_func_envi_to_npy_mp(hsi_func=calculate_ndvi)
    calculate_ndvi_mp_func(
        src_hsi_hdr_path=src_hsi_hdr_path,
        src_hsi_data_path=src_hsi_data_path,
        dst_npy_path=ndvi_dst_npy_path,
        dst_num_channels=1,
        dst_dtype=float64,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        wavelengths=wavelengths,
        zero_threshold=zero_threshold,
    )
