"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Manager
from modules import hsi_io, prosail_data

NUM_CHUNKS = 10  # Adjust based on the available computing resources

def update_progress_bar(queue, total_pixels):
    with tqdm(total=total_pixels, desc="Processing pixels") as progress_bar:
        processed_pixels = 0
        while processed_pixels < total_pixels:
            item = queue.get()
            if item is None:
                break
            processed_pixels += item
            progress_bar.update(item)

def process_chunk(start_row, end_row, hdr_path, img_path, anc_hdr_path, anc_img_path, queue):
    # Load wavelengths and ancillary data within the process
    wavelengths = hsi_io.get_wavelengths(hdr_path)
    longitude, latitude, sensor_zenith, sensor_azimuth, solar_zenith, solar_azimuth = hsi_io.get_anc_data(anc_hdr_path, anc_img_path)
    img = hsi_io.open_envi_hsi_as_np_memmap(hdr_path, img_path)
    
    # Calculate relative azimuth
    relative_azimuth = sensor_azimuth - solar_zenith
    
    # Initialize local ProsailData and result array for the chunk
    local_ps = prosail_data.ProsailData()
    x0 = (local_ps.N, local_ps.CAB, local_ps.CCX, local_ps.EWT, local_ps.LMA, local_ps.LAI, local_ps.PSOIL)
    chunk_result = np.zeros((end_row - start_row, img.shape[1], 8), dtype=np.float64)
    
    for i in range(start_row, end_row):
        for j in range(img.shape[1]):
            local_ps = prosail_data.ProsailData()
            reflectances = img[i, j]
            inversion = local_ps.fit_to_reflectances(
                (wavelengths, reflectances),
                SZA=solar_zenith[i, j],
                VZA=sensor_zenith[i, j],
                RAA=relative_azimuth[i, j]
            )
            if inversion.success:
                chunk_result[i - start_row, j] = np.array((1, local_ps.N, local_ps.CAB, local_ps.CCX, local_ps.EWT, local_ps.LMA, local_ps.LAI, local_ps.PSOIL), dtype=np.float64)
            local_ps.N, local_ps.CAB, local_ps.CCX, local_ps.EWT, local_ps.LMA, local_ps.LAI, local_ps.PSOIL = x0
            local_ps.execute()
            
            queue.put(1)  # Signal that one pixel has been processed
    
    return start_row, chunk_result

def main():
    with 
    
    # Define paths to HSI and ancillary data
    hdr_path = Path(hsi_info[project.data["hsi_config"]["hsi_name"]]["hdr"])
    img_path = Path(hsi_info[project.data["hsi_config"]["hsi_name"]]["img"])
    anc_hdr_path = Path(hsi_info[project.data["hsi_config"]["anc_name"]]["hdr"])
    anc_img_path = Path(hsi_info[project.data["hsi_config"]["anc_name"]]["img"])
    
    print("\nRunning PROSAIL Inversion ...")
    
    # Initialize result array
    img = hsi_io.open_envi_hsi_as_np_memmap(hdr_path, img_path)
    inversion_result = np.zeros((*img.shape[:2], 8), dtype=np.float64)
    img = None  # Close the memmap file to avoid resource locks
    
    # Divide the image into chunks
    num_chunks = NUM_CHUNKS  # Adjust based on the available computing resources
    chunk_size = inversion_result.shape[0] // num_chunks
    chunk_ranges = [(i * chunk_size, (i + 1) * chunk_size if i < num_chunks - 1 else inversion_result.shape[0]) for i in range(num_chunks)]
    
    # Total number of pixels
    total_pixels = inversion_result.shape[0] * inversion_result.shape[1]
    
    # Manager for shared objects
    manager = Manager()
    queue = manager.Queue()
    
    # Start a process to update the progress bar
    progress_process = Process(target=update_progress_bar, args=(queue, total_pixels))
    progress_process.start()
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, start, end, hdr_path, img_path, anc_hdr_path, anc_img_path, queue) for start, end in chunk_ranges]
        
        for future in as_completed(futures):
            start_row, chunk_result = future.result()
            inversion_result[start_row:start_row + chunk_result.shape[0], :] = chunk_result
    
    # Ensure the progress process is finished
    queue.put(None)  # Signal the progress process to exit
    progress_process.join()
    
    # Save the inversion result to a file
    inversion_result_name = "inversion_result"
    def get_inversion_result_path(inversion_result_num):
        return project.context_folder / "output" / f"{inversion_result_name}_{inversion_result_num}.npy"
    num = 0
    inversion_result_path = get_inversion_result_path(num)
    while inversion_result_path.exists():
        num += 1
        inversion_result_path = get_inversion_result_path(num)
    np.save(inversion_result_path, inversion_result)
    print(f"Inversion successfully completed! Result saved to:\n{inversion_result_path}")
    
    # end main()
    #********************************************************************************************************************************#
    #********************************************************************************************************************************#


