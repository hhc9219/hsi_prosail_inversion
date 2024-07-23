"""
Description: A template for scripts in this repository.
Email: hhc9219@rit.edu
"""

from modules.environment_manager import get_persistent_config_data

THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)

# Imports


def main():
    print("\nHello World\n")


if __name__ == "__main__":
    main()
