import numpy as np
from multiprocessing import Process, shared_memory
from typing import Callable
from .typedefs import NDArrayFloat


def float_img_to_dc(src: NDArrayFloat):
    return (src * 255).round().astype(np.uint8)


def get_img_characteristics(src: np.ndarray):
    nd = src.ndim
    if nd == 3:
        h, w, d = src.shape
        n = h * w
    elif nd == 2:
        h, d = src.shape
        n, w = h, 0
    elif nd == 1:
        d = src.shape[0]
        n, h, w = 1, 0, 0
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
        raise ValueError("src array must be either 2D or 3D.")
    num_per_part = (n + num_parts - 1) // num_parts
    start = 0
    for i in range(num_parts - 1):
        end = start + num_per_part
        yield flattened_view[start:end]
        start = end
    yield flattened_view[start:]


def split_img_by_num_parts_2d_slices(src: np.ndarray, num_parts: int):
    h, w, d, dt, nd, s, n = get_img_characteristics(src=src)
    assert nd == 2
    num_per_part = (n + num_parts - 1) // num_parts
    start = 0
    for i in range(num_parts - 1):
        end = start + num_per_part
        yield slice(start, end)
        start = end
    yield slice(start, None)


def get_num_parts_from_max_bytes(src: np.ndarray, max_bytes: int):
    h, w, d, dt, nd, s, n = get_img_characteristics(src=src)
    bytes_per_pixel = d * src.itemsize
    max_pixels_per_part = max_bytes // bytes_per_pixel
    num_parts = -(-n // max_pixels_per_part)
    return num_parts


def split_img_by_max_bytes(src: np.ndarray, max_bytes: int):
    num_parts = get_num_parts_from_max_bytes(src=src, max_bytes=max_bytes)
    yield from split_img_by_num_parts(src=src, num_parts=num_parts)


def _process_sub_chunk(
    img_func: Callable[..., np.ndarray],
    sub_chunk_slice: slice,
    src_chunk_shm_name: str,
    dst_chunk_shm_name: str,
    src_chunk_shm_a_shape: tuple,
    src_dtype: np.dtype,
    dst_chunk_shm_a_shape: tuple,
    dst_dtype: np.dtype,
    *args,
    **kwargs,
):
    # Load
    existing_src_chunk_shm = shared_memory.SharedMemory(name=src_chunk_shm_name)
    existing_dst_chunk_shm = shared_memory.SharedMemory(name=dst_chunk_shm_name)
    existing_src_chunk_shm_a = np.ndarray(
        shape=src_chunk_shm_a_shape, dtype=src_dtype, buffer=existing_src_chunk_shm.buf
    )
    existing_dst_chunk_shm_a = np.ndarray(
        shape=dst_chunk_shm_a_shape, dtype=dst_dtype, buffer=existing_dst_chunk_shm.buf
    )

    try:
        # Call img_func
        existing_dst_chunk_shm_a[sub_chunk_slice] = img_func(
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


def make_img_func_mp(img_func: Callable[..., np.ndarray]):
    from tqdm import tqdm

    def mp_img_func(
        src: np.ndarray, dst: np.ndarray, num_threads: int, max_bytes, show_progress: bool = True, *args, **kwargs
    ):
        # Free up memory that may be occupied by the dst
        if isinstance(dst, np.memmap):
            dst.flush()

        # Get src and dst image characteristics
        h_src, w_src, d_src, dt_src, nd_src, s_src, n_src = get_img_characteristics(src=src)
        h_dst, w_dst, d_dst, dt_dst, nd_dst, s_dst, n_dst = get_img_characteristics(src=dst)
        bpp_src, bpp_dst = d_src * src.itemsize, d_dst * dst.itemsize

        # Assert src and dst are both 3D
        assert nd_src == 3
        assert nd_dst == 3

        # Assert src and dst have the same height and width
        assert h_src == h_dst
        assert w_src == w_dst

        # Create a progress bar that updates with each processed chunk
        if show_progress:
            progress_bar = tqdm(total=n_src, desc="Processing Pixels")

        # Use bytes per pixel of src and dst to determine how many chunks need to be loaded into memory independently
        num_parts = get_num_parts_from_max_bytes(src=src if bpp_src > bpp_dst else dst, max_bytes=(max_bytes // 2))

        # Create shared memory arrays
        first_chunk_src = next(split_img_by_num_parts(src=src, num_parts=num_parts))
        first_chunk_dst = next(split_img_by_num_parts(src=dst, num_parts=num_parts))
        src_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_src.nbytes)
        dst_chunk_shm = shared_memory.SharedMemory(create=True, size=first_chunk_dst.nbytes)
        src_chunk_shm_a = np.ndarray(first_chunk_src.shape, dtype=dt_src, buffer=src_chunk_shm.buf)
        dst_chunk_shm_a = np.ndarray(first_chunk_dst.shape, dtype=dt_dst, buffer=dst_chunk_shm.buf)
        src_chunk_shm_name = src_chunk_shm.name
        dst_chunk_shm_name = dst_chunk_shm.name

        try:
            # Iteratively yield chunked views of the src and dst
            for chunk_src, chunk_dst in zip(
                split_img_by_num_parts(src=src, num_parts=num_parts),
                split_img_by_num_parts(src=dst, num_parts=num_parts),
            ):
                # Copy chunk_src to shared memory src
                chunk_n, _ = chunk_src.shape
                src_chunk_shm_a[:chunk_n] = chunk_src[:chunk_n]

                # Spawn processes to compute the result for each sub chunk
                processes: list[Process] = []
                for sub_chunk_slice in split_img_by_num_parts_2d_slices(src=chunk_src, num_parts=num_threads):
                    process = Process(
                        target=_process_sub_chunk,
                        args=(
                            img_func,
                            sub_chunk_slice,
                            src_chunk_shm_name,
                            dst_chunk_shm_name,
                            first_chunk_src.shape,
                            dt_src,
                            first_chunk_dst.shape,
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
                chunk_dst[:chunk_n] = dst_chunk_shm_a[:chunk_n]

                # Write changes to the chunk_dst and free memory
                if isinstance(chunk_dst, np.memmap):
                    chunk_dst.flush()

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

    return mp_img_func
