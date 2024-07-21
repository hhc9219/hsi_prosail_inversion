import colour
import numpy as np
from numpy.typing import NDArray


class ColorConverter:
    def __init__(
        self,
        wavelengths: NDArray[np.float64],
        illuminant=colour.SDS_ILLUMINANTS["D65"],
        cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    ):
        self.wavelengths = wavelengths
        self.illuminant = illuminant
        self.cmfs = cmfs

    @staticmethod
    def xyz_to_sRGB(xyz):
        return colour.XYZ_to_sRGB(xyz / 100).astype(np.float64)

    def reflectances_to_xyz(
        self,
        reflectances: NDArray[np.float64],
    ):
        sd = colour.SpectralDistribution(data=reflectances, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return xyz

    def reflectances_to_sRGB(
        self,
        reflectances: NDArray[np.float64],
    ):
        sd = colour.SpectralDistribution(data=reflectances, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return self.xyz_to_sRGB(xyz)

    @staticmethod
    def float_array_to_dc_array(float_array: NDArray[np.float64]):
        return (float_array * 255).astype(np.int64)

    def hsi_to_sRGBi(self, hsi: NDArray[np.float64]):
        sRGBi = np.apply_along_axis(func1d=self.reflectances_to_sRGB, axis=2, arr=hsi)
        return self.float_array_to_dc_array(sRGBi)


def hsi_to_sRGB(
    hsi: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(wavelengths=wavelengths, illuminant=illuminant, cmfs=cmfs)
    return color_converter.hsi_to_sRGBi(hsi=hsi)


def downsample(hsi: NDArray[np.float64], new_height: int, new_width: int):
    original_height, original_width, channels = hsi.shape
    row_step = original_height // new_height
    col_step = original_width // new_width
    resized_image = hsi[::row_step, ::col_step, :]
    resized_image = resized_image[:new_height, :new_width, :]
    return resized_image
