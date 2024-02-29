"""
A Script to invert the prosail radiative transfer model using the Nelder-Mead simplex method
on a pixel by pixel basis for an ENVI hyperspectral image.

hhc9219@rit.edu
"""

# import external dependencies
import os
import sys
import pathlib
import json
import numpy as np
import spectral.io.envi as envi
from scipy import optimize
from scipy.interpolate import interp1d

# from matplotlib import pyplot as plt

PROJECT_FOLDER = pathlib.Path(__file__).parent.resolve()
PERSISTENT_DATA_PATH = PROJECT_FOLDER / "persistent_data.json"
sys.path.append(str(PROJECT_FOLDER))
# import constants for prosail inversion process
import prosail_inv_cnsts as cnst


class ProsailBuildError(Exception):
    pass


# import prosail, attempt to build if not found
try:
    import prosail
except ImportError:
    if input("PROSAIL not found, build from source (y/n): ").strip() == "y":
        import subprocess

        PROSAIL_FOLDER = PROJECT_FOLDER / "external_packages" / "prosail"
        if not PROSAIL_FOLDER.exists():
            subp_retval = subprocess.call(["git", "clone", "https://github.com/jgomezdans/prosail", PROSAIL_FOLDER])
        CURRENT_FOLDER = os.getcwd()
        os.chdir(PROSAIL_FOLDER)
        subp_retval = subprocess.call([sys.executable, PROSAIL_FOLDER / "setup.py", "install"])
        os.chdir(CURRENT_FOLDER)
        if subp_retval != 0:
            raise ProsailBuildError("BUILD FAILURE, consider building manually.")
        sys.exit("BUILD SUCCESS, please run again.")
    else:
        sys.exit("Please build prosail to run this script")


class ImageLoadError(Exception):
    pass


# get hyperspectral image path
if PERSISTENT_DATA_PATH.exists():
    with open(PERSISTENT_DATA_PATH, "r") as f:
        persistent_data = json.load(f)
else:
    persistent_data = {
        "img_hdr_path": input("Please input the full path to the hdr file including the extension:\n").strip()
    }
    user_path_input = input("(Optional) Specify path to image data:\n").strip()
    if user_path_input == "":
        persistent_data["img_data_path"] = persistent_data["img_hdr_path"].split(".")[0]
    else:
        persistent_data["img_data_path"] = user_path_input
    with open(PERSISTENT_DATA_PATH, "w") as f:
        json.dump(persistent_data, f)
try:
    IMG_HDR_PATH = pathlib.Path(persistent_data["img_hdr_path"])
    IMG_DATA_PATH = pathlib.Path(persistent_data["img_data_path"])
    if not (IMG_HDR_PATH.exists() and IMG_DATA_PATH.exists()):
        raise ValueError
except (ValueError, KeyError):
    PERSISTENT_DATA_PATH.unlink()
    raise ImageLoadError("Image failed to load, please run again and re-enter file paths.")


class ParseEnviError(Exception):
    pass


# get wavelengths from hdr file
wavelengths_str = ""
looking_at_wavelengths = False
hdr_file = open(IMG_HDR_PATH, "r")
try:
    for line in hdr_file:
        if looking_at_wavelengths:
            wavelengths_str += line
            if line.rstrip().endswith("}"):
                break
        else:
            if line.replace(" ", "").startswith("wavelength={"):
                looking_at_wavelengths = True
                wavelengths_str += line
                if line.rstrip().endswith("}"):
                    break
    hdr_file.close()
    wavelengths_str_list = wavelengths_str.replace("\n", "").replace(" ", "")[12:-1].split(",")
    wavelengths_array = np.empty(len(wavelengths_str_list), dtype=np.float64)
    for i, v in enumerate(wavelengths_str_list):
        wavelengths_array[i] = np.float64(v)
    WAVELENGTHS = wavelengths_array
except Exception:
    hdr_file.close()
    raise ParseEnviError("Failed to parse wavelengths from hyperspectral image.")

# print(WAVELENGTHS)

# load hyperspectral image using spectral python
try:
    spy_img = envi.open(IMG_HDR_PATH, IMG_DATA_PATH)
    # assert isinstance(spy_img, envi.BsqFile)
    # use numpy memory map during processing
    np_img = spy_img.open_memmap()
except (TypeError, envi.EnviDataFileNotFoundError):
    PERSISTENT_DATA_PATH.unlink()
    raise ImageLoadError("Image failed to load, please run again and re-enter file paths.")


print("---------- STARTING PROSAIL INVERSION ----------")
# print(np_img.shape)

# loop over all of the pixels in the image
for j in range(np_img.shape[0]):
    for i in range(np_img.shape[1]):

        # print(np_img[j, i, :-1])
        # plt.plot(np_img[j, i, :-1])
        # plt.show()

        def prosail_prediction_distance(x) -> float:
            """
            Runs prosail and computes the Euclidean distance between the spectra produced
            by prosail and the spectra at the current pixel in the hyperspectral image
            """
            n, cab, ccx, ewt, lma, lai, psoil = x
            hspot = 0.5 / lai
            ps_spectrum = prosail.run_prosail(
                n=n,
                cab=cab,
                car=ccx,
                cbrown=cnst.CBP,
                cw=ewt,
                cm=lma,
                lai=lai,
                typelidf=cnst.TYPELIDF,
                lidfa=cnst.LIDFA,
                lidfb=cnst.LIDFB,
                hspot=hspot,
                psoil=psoil,
                rsoil=cnst.RSOIL,
                tts=cnst.SZA,
                tto=cnst.VZA,
                psi=cnst.RAA,
            )
            assert isinstance(ps_spectrum, np.ndarray)

            """
            perform nearest neighbor interpolation on the prosail output
            so that prosail's bands correspond to the HSI's bands
            """
            interpolate = interp1d(np.arange(400, 2501), ps_spectrum, kind="nearest")
            interp_ps_spectrum = interpolate(WAVELENGTHS[:-1])

            return np.sqrt(np.sum((interp_ps_spectrum - np_img[j, i, :-1]) ** 2))

        # use Nelder-Mead to invert the prosail parameters for the current pixel
        nelder_mead_retval = optimize.minimize(
            prosail_prediction_distance,
            x0=(cnst.N_AVG, cnst.CAB_AVG, cnst.CCX_AVG, cnst.EWT_AVG, cnst.LMA_AVG, cnst.LAI_AVG, cnst.PSOIL_AVG),
            method="Nelder-Mead",
            bounds=(
                (cnst.N_MIN, cnst.N_MAX),
                (cnst.CAB_MIN, cnst.CAB_MAX),
                (cnst.CCX_MIN, cnst.CCX_MAX),
                (cnst.EWT_MIN, cnst.EWT_MAX),
                (cnst.LMA_MIN, cnst.LMA_MAX),
                (cnst.LAI_MIN, cnst.LAI_MAX),
                (cnst.PSOIL_MIN, cnst.PSOIL_MAX),
            ),
            options={"disp": True, "adaptive": True},
        )
        print(nelder_mead_retval)

        # TODO store Nelder-Mead results once working

        break  # remove to keep looping
    break  # remove to keep looping
