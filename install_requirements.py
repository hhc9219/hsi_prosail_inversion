import sys
import platform
import subprocess
import shutil
from pathlib import Path
from modules.context_manager import Context

def main():
    with Context() as project:
        if project.data:
            print(
                "\nThe installation process has already been run.\n"
                "If you are having issues, please re-clone this repository and retry installation.\n"
                )
        else:
            get_project_data(project_context=project)
            install_requirements(project_context=project)

class SubprocessError(Exception):
    pass

class InstallVenvError(Exception):
    pass

class InstallProsailError(Exception):
    pass

def get_sys_platform(allowed_platforms: list[str] | tuple[str, ...] | str | None = None):
    """
    Returns the system platform after checking if it is an allowed_platform.
    """
    system_platform = platform.system()
    if isinstance(allowed_platforms, str):
        if system_platform != allowed_platforms:
            raise OSError(f"\nDetected system platform '{system_platform}' is not '{allowed_platforms}'.\n")
    if isinstance(allowed_platforms, (list, tuple)):
        if system_platform not in allowed_platforms:
            raise OSError(f"\nDetected system platform '{system_platform}' is not in: {allowed_platforms}.\n")
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
        raise OSError("\nPlatform is not Windows, Linux, or Darwin (Mac).\n")
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
        raise InstallVenvError("\nVirtual environment python executable was not installed at the expected location.\n")
    if not venv_pip.exists():
        raise InstallVenvError("\nVirtual environment pip executable was not installed at the expected location.\n")
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
        raise InstallProsailError("\nProsail folder was not installed at the expected location.\n")
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
        "\nInstallation was successfully completed. Please run the following command in your terminal:\n"
        f"\n{project_context.data["venv_python"]} run_inversion.py\n"
        )


if __name__ == "__main__":
    main()
