# Prosail Inversion for a ENVI Hyperspectral image

## Requirements

#### Software

* Python 3.12.1  ( May support other versions as well. )
* Windows  ( Supports other operating systems like Linux and Mac, in theory. )

#### Hardware

* 12 Thread CPU  ( Can configure to be less, see Not Enougth CPU Threads under Troubleshooting. )

#### Hyperspectral Images

The Hyperspectral images used while developing this repository may be found on the CIS Linux systems at:

```plaintext
/dirs/processing/durip/rse4949/prosail_study/run-prosail/imagery_data/site-150P/150-P_0Poles_July12_2017_run5/raw_1944_refl_plaque3.hdr
/dirs/processing/durip/rse4949/prosail_study/run-prosail/imagery_data/site-150P/150-P_0Poles_July12_2017_run5/raw_1944_refl_plaque3.img
/dirs/processing/durip/rse4949/prosail_study/run-prosail/imagery_data/site-150P/150-P_0Poles_July12_2017_run5/raw_1944_refl_plaque3_anc.hdr
/dirs/processing/durip/rse4949/prosail_study/run-prosail/imagery_data/site-150P/150-P_0Poles_July12_2017_run5/raw_1944_refl_plaque3_anc.img
```

The corresponding inversion result for this image set produced using this repository may be found at:

```plaintext
/dirs/processing/durip/hhc9219/prosail_inversion_results/agb/inversion_result_0.npy
```

## Quickstart

### process_prosail.py

Run the "process_prosail.py" script and follow the prompts to automate the installation and configuration setup. Once setup correctly, this script will begin the inversion process and provide a progress bar indicating the time to completion. Once complete, a numpy array containing the inversion result will be saved to the "output" folder.

### inversion_result_analysis.ipynb

The "inversion_result_analysis.ipynb" jupyter notebook shows an example of how to load a inversion result and displays the channels corresponding to the various parameters. Please note that in order to run this file you should set your jupyter kernel to use the virtual environment's python located in "venv".

### hsi folder

The "hsi" folder provides a convenient place to optionally store your hyperspectral images. There's nothing particualrarly important about it, but it's contents are included in the .gitignore, which will ensure potentially large hyperspectral images aren't pushed to the remote repository.

## Troubleshooting

### Potential Issues

#### Improper Configuration

Most issues will likely stem from improperly configured paths in "hsi_info.json", or improperly configured HSI or ANC names. Ensure that the paths in "hsi_info.json" properly reference the intended files, and that the HSI and ANC names you provide are keys within the "hsi_info.json" file.

Additionally, do not modify a subkey name like "hdr" or "img" within the "hsi_info.json" file, even if the corresponding file extenstion does not match.

#### Not Enougth CPU Threads

By default, "process_prosail.py" runs using 12 threads. 10 of which are used to process smaller portions, or chuncks, of the provided hyperspectral image. In the current implementation, the number threads used can be determined by 2 + NUM_CHUNKS.

NUM_CHUNKS is a global variable defined at the top of "process_prosail.py" after the initial imports. The value assigned to NUM_CHUNKS may be reduced in order to reduce the total number of required threads.

Please note that as you decrease the number of threads used, the total processing time increases.

#### Long Processing time

In order to reduce the processing time without using a smaller or different hyperspectral image, editing the fit_to_reflectances method of the ProsailData class may be the best place to start. This class is located in "prosail_inversion/prosail_data.py". Specifically I'd look into modifying the call to scipy.optimize.minimize which implements the Nelder-Mead simplex method to invert the prosail parameters for a single pixel. Adjusting parameters such as "fatol" and "ratol" could significantly decrease the processing time.

If your computer hardware allows for it, an easy speedup can be accomplished by increasing the number of threads used for processing. The process to achieve this can be inferred from the Not Enougth CPU Threads section of this document.

#### Other Issues

I'd reccomend re-cloning this repository and running the "process_prosail.py" script from scratch.

If issues persist, a more complicated issue may need to be addressed.
