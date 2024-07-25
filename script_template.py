"""
Description: A template for scripts in this repository.
Email: hhc9219@rit.edu
"""


def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    # IMPORTS

    print("\nHello World\n")


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()
