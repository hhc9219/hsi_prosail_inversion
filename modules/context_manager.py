import os
import sys
import json
from pathlib import Path
from typing import Any


class Context:
    data_filename = "data.json"

    def __init__(
        self, folder: Path | str | None = None, data_filename: str | None = None, add_to_path: bool | None = None
    ) -> None:
        do_add_to_path = add_to_path
        self.file_folder = Path(".").parent.resolve()
        if folder:
            context_folder = folder.resolve() if isinstance(folder, Path) else Path(folder).resolve()
            if context_folder.exists():
                self.context_folder = context_folder
            else:
                raise ValueError(f"The path: '{context_folder}' does not exist.")
        else:
            self.context_folder = self.file_folder
            do_add_to_path = True if add_to_path is None else add_to_path
        if do_add_to_path:
            context_folder_str = str(self.context_folder)
            if context_folder_str not in sys.path:
                sys.path.append(context_folder_str)
        data_name = data_filename if data_filename else self.data_filename
        self.data_path = self.context_folder / data_name
        self.data_path_exists = self.data_path.exists()
        if self.data_path_exists:
            with open(self.data_path, "r") as f:
                self.data: dict[str, Any] = json.load(f)
        else:
            self.data: dict[str, Any] = {}
        self.temp_data: dict[str, Any] = {}
        self.last_folder = None

    def __enter__(self):
        self.last_folder = Path(os.getcwd()).resolve()
        os.chdir(self.context_folder)
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type:ignore
        if self.last_folder:
            os.chdir(self.last_folder)
            if self.data:
                with open(self.data_path, "w") as f:
                    json.dump(self.data, f, indent=4)
            else:
                if self.data_path_exists:
                    self.data_path.unlink()
        else:
            raise RuntimeError("Could not restore the last working directory because it was never set.")

        context_folder_str = str(self.context_folder)
        if context_folder_str in sys.path:
            sys.path.remove(context_folder_str)

    @staticmethod
    def is_using_python_exe(python_exe_path: str | Path):
        """
        Checks if python is being run from the correct executable.
        """
        exe_path = python_exe_path.resolve() if isinstance(python_exe_path, Path) else Path(python_exe_path).resolve()
        return Path(sys.executable).resolve() == exe_path


def enforce_venv(file: str, data_filename: str | None = None, venv_python_data_key: str = "venv_python"):
    with Context(data_filename=data_filename) as project:
        if project.data:
            if venv_python_data_key in project.data:
                venv_python = project.data[venv_python_data_key]
                if not project.is_using_python_exe(venv_python):
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
