"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""

import os
import sys
import platform
import subprocess
import json
import shutil
import hashlib
from typing import Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Manager

NUM_CHUNKS = 10  # Adjust based on the available computing resources

def update_progress_bar(queue, total_pixels):
    from tqdm import tqdm
    with tqdm(total=total_pixels, desc="Processing pixels") as progress_bar:
        processed_pixels = 0
        while processed_pixels < total_pixels:
            item = queue.get()
            if item is None:
                break
            processed_pixels += item
            progress_bar.update(item)

def process_chunk(start_row, end_row, hdr_path, img_path, anc_hdr_path, anc_img_path, queue):
    import numpy as np
    from prosail_inversion import prosail_data, hsi_io
    
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
    import numpy as np
    from prosail_inversion import hsi_io

    # Load hyperspectral image info and update the config
    hsi_manager = HsiInfoManager(project)
    hsi_info = hsi_manager.load_hsi_info()
    hsi_manager.prompt_update_hsi_config()
    
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

"""
# Old and likely broken no multiprocessing implementation

def main():
    from prosail_inversion import prosail_data, hsi_io
    hsi_info = load_hsi_info()
    prompt_update_hsi_config()
    hsi_hdr_path = Path(hsi_info[project.data["hsi_config"]["hsi_name"]]["hdr"])
    hsi_img_path = Path(hsi_info[project.data["hsi_config"]["hsi_name"]]["img"])
    anc_hdr_path = Path(hsi_info[project.data["hsi_config"]["anc_name"]]["hdr"])
    anc_img_path = Path(hsi_info[project.data["hsi_config"]["anc_name"]]["img"])
    
    wavelengths = hsi_io.get_wavelengths(hsi_hdr_path)
    longitude, latitude, sensor_zenith, sensor_azimuth, solar_zenith, solar_azimuth = hsi_io.get_anc_data(anc_hdr_path, anc_img_path)
    img = hsi_io.open_envi_hsi_as_np_memmap(hsi_hdr_path,hsi_img_path)
    
    relative_azimuth = sensor_azimuth - solar_azimuth
    inversion_result = np.zeros((*img.shape[:2], 8), dtype=np.float64)

    ps = prosail_data.ProsailData()
    x0 = (ps.N, ps.CAB, ps.CCX, ps.EWT, ps.LMA, ps.LAI, ps.PSOIL)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ps = prosail_data.ProsailData()
            reflectances = img[i,j]
            inversion = ps.fit_to_reflectances((wavelengths, reflectances), SZA=solar_zenith[i,j], VZA=sensor_zenith[i,j], RAA=relative_azimuth[i,j])
            if inversion.success:
                inversion_result[i,j] = np.array((1, ps.N, ps.CAB, ps.CCX, ps.EWT, ps.LMA, ps.LAI, ps.PSOIL), dtype=np.float64)
            ps.N, ps.CAB, ps.CCX, ps.EWT, ps.LMA, ps.LAI, ps.PSOIL = x0
            ps.execute()
    
    np.save(str(project.context_folder / "inversion_result.npy"), inversion_result)
"""


class HsiInfoManagerError(Exception):
    pass

class SubprocessError(Exception):
    pass

class InstallVenvError(Exception):
    pass

class InstallProsailError(Exception):
    pass



class HsiInfoManager:
    def __init__(self, project: Context):
        self.project = project
        self.hsi_info_path = self.project.context_folder / "hsi_info.json"
        self.template = {
            "agb_hsi": {
                "hdr": "full_path/your_hsi.hdr",
                "img": "full_path/your_hsi.img"
            },
            "agb_anc": {
                "hdr": "full_path/your_hsi.hdr",
                "img": "full_path/your_hsi.img"
            }
        }
        self.template_hash = self._calculate_hash(self.template)
        self.venv_python = self.project.data["venv_python"]

    def _calculate_hash(self, data: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(data, indent=4).encode('utf-8')).hexdigest()

    def load_hsi_info(self) -> dict[str, dict[str, Any]]:
        try:
            if self.hsi_info_path.exists():
                if self._current_hash_matches_template():
                    self._prompt_update_template()
                    return {}
                return self._read_hsi_info()
            else:
                self._create_template()
                self._prompt_update_template()
                return {}
        except Exception as e:
            raise HsiInfoManagerError(f"Failed to load HSI info: {e}")

    def _current_hash_matches_template(self) -> bool:
        with open(self.hsi_info_path, "rb") as f:
            current_hash = self._calculate_hash(json.load(f))
        return current_hash == self.template_hash

    def _prompt_update_template(self):
        print(f"\nPlease update the template in the file:\n{self.hsi_info_path}\n\nOnce complete, run this file again:\n{self.venv_python} process_prosail.py\n")
        sys.exit(0)

    def _read_hsi_info(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.hsi_info_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise HsiInfoManagerError(f"Failed to read HSI info: {e}")

    def _create_template(self):
        try:
            with open(self.hsi_info_path, "w") as f:
                json.dump(self.template, f, indent=4)
        except Exception as e:
            raise HsiInfoManagerError(f"Failed to create template: {e}")

    def prompt_update_hsi_config(self):
        try:
            if "hsi_config" in self.project.data:
                use_last_config = input(f"Would you like to use the last specified configuration:\nHSI Name: {self.project.data['hsi_config']['hsi_name']}\nANC Name: {self.project.data['hsi_config']['anc_name']}\ny/n? : ")
                if use_last_config.lower() == 'y':
                    return
                del self.project.data["hsi_config"]

            self._configure_hsi()
        except Exception as e:
            raise HsiInfoManagerError(f"Failed to update HSI config: {e}")

    def _configure_hsi(self):
        self.project.data["hsi_config"] = {}
        print("Please configure the HSI and ANC names (both should be keys specified in the hsi_info.json file):")
        self.project.data["hsi_config"]["hsi_name"] = input("HSI Name: ")
        self.project.data["hsi_config"]["anc_name"] = input("ANC Name: ")
        

def get_sys_platform(allowed_platforms: list[str] | tuple[str, ...] | str | None = None):
    """
    Returns the system platform after checking if it is an allowed_platform.
    """
    system_platform = platform.system()
    if isinstance(allowed_platforms, str):
        if system_platform != allowed_platforms:
            raise OSError(f"Detected system platform '{system_platform}' is not '{allowed_platforms}'.")
    if isinstance(allowed_platforms, (list, tuple)):
        if system_platform not in allowed_platforms:
            raise OSError(f"Detected system platform '{system_platform}' is not in: {allowed_platforms}.")
    return system_platform


def get_venv_folder(project_context: Context):
    """
    Returns the virtual environment's folder path regardless of whether it exists or not.
    """
    default_venv_folder = project_context.context_folder / "venv"
    if default_venv_folder.exists():
        return str(default_venv_folder)
    for venv_name in (".venv", "env", ".env"):
        venv_folder = project_context.context_folder / venv_name
        if venv_folder.exists():
            return str(venv_folder)
    return str(default_venv_folder)


def get_venv_python(project_context: Context):
    """
    Returns the virtual environment's python executable regardless of whether it exists or not.
    """
    venv_folder = Path(project_context.temp_data["venv_folder"])
    system_platform = project_context.data["system_platform"]
    if system_platform == "Windows":
        venv_python = venv_folder / "Scripts" / "python.exe"
    elif system_platform == "Linux" or system_platform == "Darwin":
        venv_python = venv_folder / "bin" / "python"
    else:
        raise OSError("Platform is not Windows, Linux, or Darwin (Mac).")
    return str(venv_python)


def get_venv_pip(project_context: Context):
    """
    Returns the virtual environment's pip executable regardless of whether it exists or not.
    """
    venv_python = Path(project_context.data["venv_python"])
    venv_pip = venv_python.parent / ("pip" + venv_python.suffix)
    return str(venv_pip)


def get_prosail_folder(project_context: Context):
    """
    Returns prosail's folder path regardless of whether it exists or not.
    """
    prosail_folder = project_context.context_folder / "external_packages" / "prosail"
    return str(prosail_folder)


def get_project_data(project_context: Context):
    """
    Updates the project's persistent and temporary data for installation.
    """
    project_context.data["system_platform"] = get_sys_platform(("Windows", "Linux", "Darwin"))
    project_context.temp_data["venv_folder"] = get_venv_folder(project_context=project_context)
    project_context.data["venv_python"] = get_venv_python(project_context=project_context)
    project_context.temp_data["venv_pip"] = get_venv_pip(project_context=project_context)
    project_context.temp_data["prosail_folder"] = get_prosail_folder(project_context=project_context)


def run_subprocess(command: str):
    """
    Executes a command in a subprocess and raises a detailed exception if the command fails.
    """
    try:
        return subprocess.run(
            command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise SubprocessError(
            f"\nCommand execution failed with return code: {e.returncode}.\n"
            "Command:\n"
            f"    {command}\n"
            "Error Message:\n"
            f"    {e.stderr}"
        )
    except Exception as e:
        raise SubprocessError(
            f"\nCommand execution failed.\n" "Command:\n" f"    {command}\n" "Exception:\n" f"    {e}"
        )


def install_venv(project_context: Context):
    """
    Installs a virtual environment in the project folder
    with the necessary pypi dependencies from requirements.txt
    """
    venv_folder = Path(project_context.temp_data["venv_folder"])
    if not venv_folder.exists():
        command = f"{sys.executable} -m venv {venv_folder}"
        try:
            run_subprocess(command)
        except SubprocessError as e:
            raise InstallVenvError(f"{e}")
    venv_python = Path(project_context.data["venv_python"])
    venv_pip = Path(project_context.temp_data["venv_pip"])
    if not venv_python.exists():
        raise InstallVenvError("Virtual environment python executable was not installed at the expected location.")
    if not venv_pip.exists():
        raise InstallVenvError("Virtual environment pip executable was not installed at the expected location.")
    command = f"{venv_pip} install -r {project_context.context_folder / "requirements.txt"}"
    try:
        run_subprocess(command)
    except SubprocessError as e:
        raise InstallVenvError(f"{e}")


def install_prosail(project_context: Context):
    """
    Installs prosail from source in the virtual environment.
    """
    prosail_folder = Path(project_context.temp_data["prosail_folder"])
    if prosail_folder.exists():
        shutil.rmtree(prosail_folder)
    command = f"git clone https://github.com/jgomezdans/prosail.git {prosail_folder}"
    try:
        run_subprocess(command)
    except SubprocessError as e:
        raise InstallProsailError(f"{e}")
    if not prosail_folder.exists():
        raise InstallProsailError("Prosail folder was not installed at the expected location.")
    command = f"{project_context.data["venv_python"]} {prosail_folder / "setup.py"} install"
    with Context(prosail_folder):
        try:
            run_subprocess(command)
        except SubprocessError as e:
            raise InstallProsailError(f"{e}")


def install_requirements(project_context: Context):
    """
    Installs the virtual environment and all necessary dependencies.
    """
    print("Installing requirements... (This could take a few minutes.)")
    install_venv(project_context)
    install_prosail(project_context)
    hsi_folder = project_context.context_folder / "hsi"
    output_folder =project_context.context_folder / "output"
    hsi_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(
        f"""
        Installation was successfully completed. Please run the following command in your terminal:

        {project.data["venv_python"]} process_prosail.py

        """
        )


def in_venv(project_context: Context):
    """
    Checks if python is being run from the correct virtual environment.
    """
    return Path(sys.executable).resolve() == Path(project_context.data["venv_python"]).resolve()


if __name__ == "__main__":
    with Context() as project:
        if project.data:
            if in_venv(project):
                main()
            else:
                print(
                    f"""
                    You are using the incorrect python executable, please run the following command in your terminal:

                    {project.data["venv_python"]} process_prosail.py

                    """
                    )
        else:
            get_project_data(project_context=project)
            install_requirements(project_context=project)
