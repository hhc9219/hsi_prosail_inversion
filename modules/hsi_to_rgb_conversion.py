import colour
import numpy as np
from typing import Any
from .img_processing import float_img_to_dc, make_img_func_mp

NDArrayFloat = np.ndarray[Any, np.dtype[np.float32 | np.float64]]


class ColorConverter:
    def __init__(
        self,
        original_wavelengths: NDArrayFloat,
        wavelengths_resample_interval: int | None = 1,
        illuminant=colour.SDS_ILLUMINANTS["D65"],
        cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    ):
        self.original_wavelengths = original_wavelengths

        self.wavelengths = (
            np.arange(
                round(original_wavelengths[0]),
                round(original_wavelengths[-1]) + wavelengths_resample_interval,
                wavelengths_resample_interval,
                dtype=np.int64,
            )
            if wavelengths_resample_interval is not None
            else original_wavelengths.astype(np.int64)
        )
        self.do_resample_reflectances = True if wavelengths_resample_interval is not None else False

        self.illuminant = illuminant
        self.cmfs = cmfs

    def resample_reflectances(self, reflectances: NDArrayFloat):
        return np.interp(self.wavelengths, self.original_wavelengths, reflectances)

    @staticmethod
    def xyz_to_sRGB(xyz: NDArrayFloat):

        return colour.XYZ_to_sRGB(xyz / 100).astype(xyz.dtype)

    def reflectances_to_xyz(
        self,
        reflectances: NDArrayFloat,
    ):
        refl = self.resample_reflectances(reflectances) if self.do_resample_reflectances else reflectances
        sd = colour.SpectralDistribution(data=refl, domain=self.wavelengths)
        xyz = colour.sd_to_XYZ(sd, self.cmfs, self.illuminant)
        return xyz

    def reflectances_to_sRGB_float(
        self,
        reflectances: NDArrayFloat,
    ) -> NDArrayFloat:
        xyz = self.reflectances_to_xyz(reflectances=reflectances)
        return self.xyz_to_sRGB(xyz)  # type: ignore

    def reflectances_to_sRGB(self, reflectances: NDArrayFloat):
        sRGB = self.reflectances_to_sRGB_float(reflectances=reflectances)
        return float_img_to_dc(sRGB)

    def hsi_to_sRGB_float(self, hsi: NDArrayFloat):
        if hsi.ndim == 3:
            ax = 2
        elif hsi.ndim == 2:
            ax = 1
        else:
            raise ValueError("src array must be either 2D or 3D.")
        sRGB = np.apply_along_axis(func1d=self.reflectances_to_sRGB_float, axis=ax, arr=hsi)
        return sRGB

    def hsi_to_sRGB(self, hsi: NDArrayFloat):
        sRGB = self.hsi_to_sRGB_float(hsi)
        return float_img_to_dc(sRGB)


def hsi_to_sRGB(
    hsi: NDArrayFloat,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
):
    color_converter = ColorConverter(
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
    return color_converter.hsi_to_sRGB(hsi=hsi)


def hsi_to_sRGB_mp(
    hsi_src: np.ndarray,
    rgb_dst: np.ndarray,
    original_wavelengths: NDArrayFloat,
    wavelengths_resample_interval: int | None = 1,
    illuminant=colour.SDS_ILLUMINANTS["D65"],
    cmfs=colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
    num_threads: int = 2,
    max_bytes=int(0.5e9),
    show_progress=True,
):
    hsi_to_sRGB_func = make_img_func_mp(hsi_to_sRGB)
    hsi_to_sRGB_func(
        src=hsi_src,
        dst=rgb_dst,
        num_threads=num_threads,
        max_bytes=max_bytes,
        show_progress=show_progress,
        original_wavelengths=original_wavelengths,
        wavelengths_resample_interval=wavelengths_resample_interval,
        illuminant=illuminant,
        cmfs=cmfs,
    )
