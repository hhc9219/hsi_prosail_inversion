"""
This file defines ProsailData, a class which assists in managing the parameters that dictate prosail's reflectance result.
Also, it provides inversion capabilities to determine the prosail parameters for a given reflectance.

hhc9219@rit.edu
"""

import prosail
import numpy as np
from typing import Any
from scipy import optimize
from .dynamic_data import DynamicData
from .typedefs import NDArrayFloat


class ProsailData(DynamicData):
    def __init__(
        self,
        # PROSPECT
        N_MIN=1,
        N_MAX=1.25,
        CAB_MIN=10,
        CAB_MAX=101,
        EWT_MIN=0.0015,
        EWT_MAX=0.002,
        LMA_MIN=0.002,
        LMA_MAX=0.1,
        CBP=0.0005,
        CCX_MIN=7,
        CCX_MAX=9,
        # SAIL
        LAI_MIN=1,
        LAI_MAX=10,
        PSOIL_MIN=0.2,
        PSOIL_MAX=1,
        RSOIL=1,
        TYPELIDF=1,
        LIDFA=-1,
        LIDFB=0,
        SZA=0,
        VZA=0,
        RAA=0,
        FACTOR="HDR",
    ):
        super().__init__()
        self.N_MIN = N_MIN
        self.N_MAX = N_MAX
        self.CAB_MIN = CAB_MIN
        self.CAB_MAX = CAB_MAX
        self.EWT_MIN = EWT_MIN
        self.EWT_MAX = EWT_MAX
        self.LMA_MIN = LMA_MIN
        self.LMA_MAX = LMA_MAX
        self.CBP = CBP
        self.CCX_MIN = CCX_MIN
        self.CCX_MAX = CCX_MAX
        self.LAI_MIN = LAI_MIN
        self.LAI_MAX = LAI_MAX
        self.PSOIL_MIN = PSOIL_MIN
        self.PSOIL_MAX = PSOIL_MAX
        self.RSOIL = RSOIL
        self.TYPELIDF = TYPELIDF
        self.LIDFA = LIDFA
        self.LIDFB = LIDFB
        self.SZA = SZA
        self.VZA = VZA
        self.RAA = RAA
        self.FACTOR = FACTOR
        self.N = None
        self.CAB = None
        self.EWT = None
        self.LMA = None
        self.CCX = None
        self.LAI = None
        self.PSOIL = None
        self.HSPOT = None
        avg = lambda a, b: 0.5 * (a + b)
        self.set_funcs(
            N=lambda N_MIN, N_MAX: avg(N_MIN, N_MAX),
            CAB=lambda CAB_MIN, CAB_MAX: avg(CAB_MIN, CAB_MAX),
            EWT=lambda EWT_MIN, EWT_MAX: avg(EWT_MIN, EWT_MAX),
            LMA=lambda LMA_MIN, LMA_MAX: avg(LMA_MIN, LMA_MAX),
            CCX=lambda CCX_MIN, CCX_MAX: avg(CCX_MIN, CCX_MAX),
            LAI=lambda LAI_MIN, LAI_MAX: avg(LAI_MIN, LAI_MAX),
            PSOIL=lambda PSOIL_MIN, PSOIL_MAX: avg(PSOIL_MIN, PSOIL_MAX),
        )
        self.execute()
        self.set_funcs(HSPOT=lambda LAI: 0.5 / LAI)
        self.execute()

    def run_prosail(self) -> tuple[NDArrayFloat | Any, ...]:
        reflectances = prosail.run_prosail(
            n=self.N,
            cab=self.CAB,
            car=self.CCX,
            cbrown=self.CBP,
            cw=self.EWT,
            cm=self.LMA,
            lai=self.LAI,
            typelidf=self.TYPELIDF,
            lidfa=self.LIDFA,
            lidfb=self.LIDFB,
            hspot=self.HSPOT,
            psoil=self.PSOIL,
            rsoil=self.RSOIL,
            tts=self.SZA,
            tto=self.VZA,
            psi=self.RAA,
            factor=self.FACTOR,
        )
        wavelengths = np.arange(400, 2501, dtype=np.float64)
        return wavelengths, reflectances

    def reflectance_rmse_residual(
        self,
        wavelengths: NDArrayFloat,
        reflectances: NDArrayFloat,
        SZA: float | None = None,
        VZA: float | None = None,
        RAA: float | None = None,
    ):
        if SZA:
            self.SZA = SZA
        if VZA:
            self.VZA = VZA
        if RAA:
            self.RAA = RAA
        prosail_wavelengths, prosail_reflectances = self.run_prosail()
        interp_reflectances = np.interp(wavelengths, prosail_wavelengths, prosail_reflectances)
        return np.sqrt(np.mean((interp_reflectances - reflectances) ** 2))

    def fit_to_reflectances(
        self,
        wavelengths: NDArrayFloat,
        reflectances: NDArrayFloat,
        SZA: float | None = None,
        VZA: float | None = None,
        RAA: float | None = None,
    ):
        if SZA:
            self.SZA = SZA
        if VZA:
            self.VZA = VZA
        if RAA:
            self.RAA = RAA

        def fun(x, *args):

            self.N, self.CAB, self.CCX, self.EWT, self.LMA, self.LAI, self.PSOIL = x
            self.execute()
            return self.reflectance_rmse_residual(wavelengths, reflectances)

        result = optimize.minimize(
            fun=fun,
            x0=np.array((self.N, self.CAB, self.CCX, self.EWT, self.LMA, self.LAI, self.PSOIL), dtype=np.float64),
            method="Nelder-Mead",
            bounds=(
                (self.N_MIN, self.N_MAX),
                (self.CAB_MIN, self.CAB_MAX),
                (self.CCX_MIN, self.CCX_MAX),
                (self.EWT_MIN, self.EWT_MAX),
                (self.LMA_MIN, self.LMA_MAX),
                (self.LAI_MIN, self.LAI_MAX),
                (self.PSOIL_MIN, self.PSOIL_MAX),
            ),
            options={"adaptive": True, "fatol": 0.01, "xatol": 2.5},
        )
        if result.success:
            self.N, self.CAB, self.CCX, self.EWT, self.LMA, self.LAI, self.PSOIL = result.x
            self.execute()
        return result.success


"""
Example val mapping:

prosail.run_prosail(
    n=N,
    cab=CAB,
    car=CCX,
    cbrown=CBP,
    cw=EWT,
    cm=LMA,
    lai=LAI,
    typelidf=TYPELIDF,
    lidfa=LIDFA,
    lidfb=LIDFB,
    hspot=hspot,
    psoil=PSOIL,
    rsoil=RSOIL,
    tts=SZA,
    tto=VZA,
    psi=RAA,
)
"""
