"""
A file which defines constants that are fed to the Nelder-Mead optimization function.
These values are adapted from Rehman Eon's paper on above ground biomass.

hhc9219@rit.edu
"""

import numpy as np

# PROSPECT

N_MIN = 1
N_MAX = 1.25
N_AVG = 0.5 * (N_MIN + N_MAX)

CAB_MIN = 10
CAB_MAX = 101
CAB_AVG = 0.5 * (CAB_MIN + CAB_MAX)

EWT_MIN = 0.001
EWT_MAX = 0.02
EWT_AVG = 0.5 * (EWT_MIN + EWT_MAX)

LMA_MIN = 20
LMA_MAX = 1000
LMA_AVG = 0.5 * (LMA_MIN + LMA_MAX)

CBP = 0

CCX_MIN = 1
CCX_MAX = 20
CCX_AVG = 0.5 * (CCX_MIN + CCX_MAX)

# SAIL

LAI_MIN = 1
LAI_MAX = 10
LAI_AVG = 0.5 * (LAI_MIN + LAI_MAX)

HSPOT = 0.5 / ((LAI_MIN + LAI_MAX) / 2)

PSOIL_MIN = 0.2
PSOIL_MAX = 1
PSOIL_AVG = 0.5 * (PSOIL_MIN + PSOIL_MAX)

RSOIL = np.ones(2101)

TYPELIDF = 1
LIDFA = 1
LIDFB = 0

SZA = 0
VZA = 0
RAA = 0

"""
Example parameter mapping:

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
