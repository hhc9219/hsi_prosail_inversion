# PROSAIL Inversion for ENVI Hyperspectral Imagery

### Getting Started, Running with Windows Powershell:

Navigate to the project's root directory `hsi_prosail_inversion` and create a python virtual environment:

```powershell
PS C:\hsi_prosail_inversion> python -m venv venv
```

Activate the virtual environment:

```powershell
PS C:\hsi_prosail_inversion> venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
(venv) PS C:\hsi_prosail_inversion> pip install -r requirements.txt
```

#### Running process_prosail.py:

```powershell
(venv) PS C:\hsi_prosail_inversion> python process_prosail.py
```

The first time this script is run you will be prompted to build prosail, type "y" and press ENTER:

```plaintext
PROSAIL not found, build from source (y/n): y
```

You should recieve a `BUILD SUCCESS` message and the program will exit. Run the same script again, and follow the prompts to specify the path to your ENVI hyperspectral image file.


**The entire setup process is now complete.**

The script should begin the PROSAIL inversion process, as indicated by the message:

```plaintext
---------- STARTING PROSAIL INVERSION ----------
```

---



### Troubleshooting

If issues arise, first try deleting the `persistent_data.json` file before re-running the script.

If issues continue, try deleting both the `venv` folder and the `persistent_data.json` file. Now, follow the "Getting Started" instructions again to restore the virtual environment.
