# Prosail Inversion for a ENVI Hyperspectral image

## Requirements

#### Software

* Python 3.12.1  ( May support other versions as well. )
* Windows  ( Supports other operating systems like Linux and Mac, in theory. )

#### Hardware

* Multiple Thread CPU ( Minimum of 3 )
* Large RAM Capacity

#### Hyperspectral Images

The Hyperspectral images used while developing this repository may be found on the CIS Linux systems at:

```plaintext
/dirs/processing/durip/nn6721/soil_organic_content/hog_island_marsh/uas/layer_stacked_marsh_1_2/
```

## Quickstart

### install_requirements.py

Run this script with your global python interpreter to automate the installation process.

### run_prosail_inversion.py

Run this script using the automatically created virtual environment and follow the dialogue to invert the PROSAIL radiative transfer model using the Nelder Mead Simplex method for an ENVI hyperspectral image.

### inversion_result_analysis.ipynb

The "inversion_result_analysis.ipynb" jupyter notebook shows an example of how to load a inversion result and displays the channels corresponding to the various parameters. Please note that in order to run this file you should set your jupyter kernel to use the virtual environment's python located in "venv".

### run_hsi_to_rgb.py

This script provides the capability to convert a ENVI hyperspectral image to a sRGB image. By default this script uses a D65 illuminant and the CIE 1931 2 Degree Standard Observer CMFs.

### output folder

This folder will be populated with script outputs and results.

### hsi folder

The "hsi" folder provides a convenient place to optionally store your hyperspectral images. There's nothing particualrarly important about it, but it's contents are included in the .gitignore, which will ensure potentially large hyperspectral images aren't pushed to the remote repository.

## Troubleshooting

### Potential Issues

#### Hardware Limitations and Memory Leaks

If you encounter issues with hardware limitations like cpu usage or memory, I reccomend decreasing the values in "resource_config.json" Please note that as you decrease the number of threads used, the total processing time increases.

#### Improper HSI Configuration

Most issues will likely stem from improperly configured paths in "hsi_config.json", or improperly configured HSI names. Ensure that the paths in "hsi_config.json" properly reference the intended files, and that the HSI names you provide are keys within the "hsi_config.json" file.

Additionally, do not modify a subkey name like "hdr" or "img" within the "hsi_config.json" file, even if the corresponding file extenstion does not match.

#### Long Processing time

If your computer hardware allows for it, an easy speedup can be accomplished by increasing the number of threads and/or the amount of memory used for processing. This can be achieved by increasing the values within "resource_config.json" to a number that doesn't exceed your system's hardware resources.

In order to reduce the processing time without using a smaller or different hyperspectral image, editing the fit_to_reflectances method of the ProsailData class may be the best place to start. This class is located in "modules/prosail_data.py". Specifically I'd look into modifying the call to scipy.optimize.minimize which implements the Nelder-Mead simplex method to invert the prosail parameters for a single pixel. Adjusting parameters such as "fatol" and "ratol" could significantly decrease the processing time.

#### Other Issues

I'd reccomend re-cloning this repository and running the "install_requirements.py" script from scratch.

If issues persist, a more complicated problem may need to be addressed.

## Showcase

### run_hsi_to_rgb.py

#### Results for the 2019 Hog Island Field Campaign

![hog_island_2019_0](docs/pictures/hog_island_2019_0.png)
![hog_island_2019_1](docs/pictures/hog_island_2019_1.png)
