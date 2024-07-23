import sys
from pathlib import Path
from .context_manager import Context


def is_using_python_exe(python_exe_path: str | Path):
    """
    Checks if python is being run from the correct executable.
    """
    exe_path = python_exe_path.resolve() if isinstance(python_exe_path, Path) else Path(python_exe_path).resolve()
    return Path(sys.executable).resolve() == exe_path


def enforce_venv(
    file: str, data_filename: str | None = "environment_config.json", venv_python_data_key: str = "venv_python"
):
    with Context(file=file, data_filename=data_filename) as project:
        if project.data:
            if venv_python_data_key in project.data:
                venv_python = project.data[venv_python_data_key]
                if not is_using_python_exe(venv_python):
                    print(
                        "\nYou are using the incorrect python executable to run this script.\n"
                        "Please use the following command to run this script correctly:\n"
                        f"\n{venv_python} {Path(file).name}\n"
                    )
                    sys.exit(0)
            else:
                raise ValueError(
                    f"\n{venv_python_data_key} was not identified as a key in {data_filename}\n"
                    "Please ensure installation is complete by running install_requirements.py\n"
                )
        else:
            raise FileNotFoundError(
                f"\n{data_filename} was not found.\n"
                "Please ensure installation is complete by running install_requirements.py\n"
            )


def get_project_folder(file:str):
    with Context(file=file) as project:
        project_folder = project.context_folder
    return project_folder


def get_resource_values(file:str) -> tuple[int, float]:
    with Context(file=file, data_filename="resource_config.json") as resource_config:
        if not resource_config.data:
            cpu_thread_count = int(input("|SET| resource_config.json <- cpu_thread_count = "))
            memory_GB = float(input("|SET| resource_config.json <- memory_GB = "))
            resource_config.data["cpu_thread_count"] = cpu_thread_count
            resource_config.data["memory_GB"] = memory_GB
        resource_values = resource_config.data["cpu_thread_count"], resource_config.data["memory_GB"]
    return resource_values


def get_hsi_config(file:str):
    no_data = False
    with Context(file=file, data_filename="hsi_config.json") as hsi_config:
        if hsi_config.data:
            if "hsi_0_abv_name" in hsi_config.data:
                print("\nThe template in the file, 'hsi_config.json' must be modified in order to complete setup.")
                no_data = True
            else:      
                hsi_config_info = hsi_config.data
        else:
            hsi_config.data = {
                "hsi_0_abv_name": {"hdr": "/full/path/to/your/hsi_0.hdr", "img": "/full/path/to/your/hsi_0.img"},
                "hsi_1_abv_name": {"hdr": "/full/path/to/your/hsi_1.hdr", "img": "/full/path/to/your/hsi_1.img"},
            }
            print("\nPlease edit the template in the newly created file, 'hsi_config.json' in order to complete setup.")
            no_data = True
    if no_data:
        print(f"Full path: {hsi_config.context_folder / "hsi_config.json"}\n")
        sys.exit(0)
    return hsi_config_info

def get_persistent_config_data(file:str):
    enforce_venv(file)
    threads, memory = get_resource_values(file)
    hsi_config = get_hsi_config(file)
    project_folder = get_project_folder(file)
    output_folder = project_folder / "output"
    print("\nPersistent configuration data was successfully loaded.\n")
    return threads, memory, hsi_config, project_folder, output_folder
