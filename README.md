# Prosail Inversion for a ENVI Hyperspectral image

## Requirements

#### Software

* Windows ( Supports other operating systems like Linux and Mac, in theory. )
* Python 3.12.1 ( May support other versions as well. )
* Git

  Please ensure Python and Git are accessible in your system's path.

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

### plot_inversion_results.py

The "plot_inversion_results.py" script loads an inversion result and creates figures in the output folder which depict the values obtained for the inversion parameters.

### run_hsi_to_rgb.py

This script provides the capability to convert a ENVI hyperspectral image to a sRGB image. By default this script uses a D65 illuminant and the CIE 1931 2 Degree Standard Observer CMFs.

### output folder

This folder will be populated with script outputs and results.

### hsi folder

The "hsi" folder provides a convenient place to optionally store your hyperspectral images. There's nothing particualrarly important about it, but it's contents are included in the .gitignore, which will ensure potentially large hyperspectral images aren't pushed to the remote repository.

### vp_mapping.ipynb

**Important:** Check the info at the top of this file in order to run, it does not use the same python environment that the install script sets up.

This python notebook contains a variety of processing steps to map the validation points or ground truths for the 2019 hog island field campaign to pixel locations within the uas hyperspectral images. Also, this notebook consolidates the solar and sensor geometry into an image like format containg spatial information for the validation points along with the corresponding geometries. These files are saved as "vp_geo_hog_island_2019_0.npy" and "vp_geo_hog_island_2019_1.npy" by the program. The channels of these arrays are: solar zenith, sensor zenith, and relative azimuth. The values are in degrees and NaNs are present where the geometry data has not been processed yet.

## Troubleshooting

### Potential Issues

#### Hardware Limitations and Memory Leaks

If you encounter issues with hardware limitations like cpu usage or memory, I reccomend decreasing the values in "resource_config.json" Please note that as you decrease the number of threads used, the total processing time increases.

#### Improper HSI Configuration

Most issues will likely stem from improperly configured paths in "hsi_config.json", or improperly configured HSI names. Ensure that the paths in "hsi_config.json" properly reference the intended files, and that the HSI names you provide are keys within the "hsi_config.json" file.

Additionally, do not modify a subkey name like "hdr" or "img" within the "hsi_config.json" file, even if the corresponding file extenstion does not match.

#### Long Processing time

If your computer hardware allows for it, an easy speedup can be accomplished by increasing the number of threads and/or the amount of memory used for processing. This can be achieved by increasing the values within "resource_config.json" to a number that doesn't exceed your system's hardware resources.

In order to reduce the processing time without using a smaller or different hyperspectral image, editing the fit_to_reflectances method of the ProsailData class may be the best place to start. This class is located in "modules/prosail_data.py". Specifically I'd look into modifying the call to scipy.optimize.minimize which implements the Nelder-Mead simplex method to invert the prosail parameters for a single pixel. Adjusting parameters such as "atol_rmse_residual" and "atol_wavelength" could significantly decrease the processing time.

#### Other Issues

I'd reccomend re-cloning this repository and running the "install_requirements.py" script from scratch.

If issues persist, a more complicated problem may need to be addressed.

## Showcase

### RGB Images with Validation Points
run_hsi_to_rgb.py & vp_mapping.ipynb

##### Results for the 2019 Hog Island Field Campaign

![vp_map_hog_island_2019_0](docs/pictures/vp_map_hog_island_2019_0.png)
![vp_map_hog_island_2019_1](docs/pictures/vp_map_hog_island_2019_1.png)

### NDVI (Normalized Difference Vegetation Index)
run_prosail_inversion.py

##### Results for the 2019 Hog Island Field Campaign

![ndvi_hog_island_2019_0](docs/pictures/ndvi_hog_island_2019_0.png)
![ndvi_hog_island_2019_1](docs/pictures/ndvi_hog_island_2019_1.png)
