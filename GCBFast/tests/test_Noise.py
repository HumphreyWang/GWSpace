#!/usr/bin/env python
#-*- coding: utf-8 -*-  
#==================================
# File Name: test_Noise.py
# Author: ekli
# Mail: lekf123@163.com
# Created Time: 2023-09-05 15:03:28
#==================================

import numpy as np
#from active_python_path import csgwsim as gws
#from csgwsim import TianQinNoise, LISANoise, noise_XYZ, noise_AET
from pyNoise import TianQinNoise, LISANoise, noise_XYZ, noise_AET
import matplotlib.pyplot as plt
#from Constants import PI, C_SI
#PI = gws.Constants.PI
#C_SI = gws.Constants.C_SI
PI = np.pi
C_SI = 299792458


if __name__ == "__main__":
    freq_ = np.logspace(-5, 0, 1001)
    la = LISANoise()
    lisa_sa, lisa_sp = la.noises(freq_)

    tq = TianQinNoise()
    tq_sa, tq_sp = tq.noises(freq_)

    LA, LE, LT = noise_AET(freq_, lisa_sa, lisa_sp, la.armL)
    TA, TE, TT = noise_AET(freq_, tq_sa, tq_sp, tq.armL)

    LX, LXY = noise_XYZ(freq_, lisa_sa, lisa_sp, la.armL, includewd=1.2)
    TX, TXY = noise_XYZ(freq_, tq_sa, tq_sp, tq.armL)
    # =======================================
    plt.figure()
    plt.title("Sensitivity of TianQin and LISA---with displacement")

    plt.loglog(freq_, la.sensitivity(freq_, includewd=1.3), label="LISA")
    plt.loglog(freq_, tq.sensitivity(freq_), label="TianQin")

    plt.legend(loc="best")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    # =======================================
    # Acc and Pos noise from Marsat 2003.00357: eq: A17a and A17b
    M_sop = lambda f: (2*PI*f/C_SI)**2*(8.9**2+1.7**2+4)*1e-24
    M_spm = lambda f: ((2*PI*f*C_SI)**(-2)*9e-30*(1+36*(1e-8/f**2+(3e-5/f)**10))
                       + 0.25*1.7e-12**2*(2*PI*f/C_SI)**2)

    plt.figure()
    plt.title("Acc and Pos noise model---relativeFrequency")

    plt.loglog(freq_, lisa_sa, ':r', label="LISA Sa")
    plt.loglog(freq_, lisa_sp, '--r', label="LISA Sp")

    plt.plot(freq_, tq_sa, "-b", label="TianQin Sa")
    plt.plot(freq_, tq_sp, "--b", label="TianQin Sp")

    plt.loglog(freq_, M_spm(freq_), '-g', label="Marsat Sa")
    plt.loglog(freq_, M_sop(freq_), '--g', label="Marsat Sp")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for A E T channel---relativeFrequency")

    plt.loglog(freq_, LA, "-", label="LISA AE")
    plt.loglog(freq_, LT, "-.", label="LISA T")

    plt.loglog(freq_, TA, "--", label="TianQin AE")
    plt.loglog(freq_, TT, "-.", label="TianQin T")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for X and X-Y channel---relativeFrequency")

    plt.loglog(freq_, LX, "-", label="LISA X")
    plt.loglog(freq_, LXY, "-.", label="LISA XY")

    plt.loglog(freq_, TX, "--", label="TianQin X")
    plt.loglog(freq_, TXY, "-.", label="TianQin XY")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    # **************************************
    la = LISANoise()
    lisa_sa, lisa_sp = la.noises(freq_, unit="displacement")

    tq = TianQinNoise()
    tq_sa, tq_sp = tq.noises(freq_, unit="displacement")

    LA, LE, LT = noise_AET(freq_, lisa_sa, lisa_sp, la.armL, includewd=1.2)
    TA, TE, TT = noise_AET(freq_, tq_sa, tq_sp, tq.armL)

    LX, LXY = noise_XYZ(freq_, lisa_sa, lisa_sp, la.armL)
    TX, TXY = noise_XYZ(freq_, tq_sa, tq_sp, tq.armL)

    # =======================================
    plt.figure()
    plt.title("Acc and Pos noise model---displacement")

    plt.loglog(freq_, lisa_sa, ':r', label="LISA Sa")
    plt.loglog(freq_, lisa_sp, '--r', label="LISA Sp")

    plt.plot(freq_, tq_sa, "-b", label="TianQin Sa")
    plt.plot(freq_, tq_sp, "--b", label="TianQin Sp")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for A E T channel---displacement")

    plt.loglog(freq_, LA, "-", label="LISA AE")
    plt.loglog(freq_, LT, "-.", label="LISA T")

    plt.loglog(freq_, TA, "--", label="TianQin AE")
    plt.loglog(freq_, TT, "-.", label="TianQin T")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    # =======================================
    plt.figure()
    plt.title("PSD of noise for X and X-Y channel---displacement")

    plt.loglog(freq_, LX, "-", label="LISA X")
    plt.loglog(freq_, LXY, "-.", label="LISA XY")

    plt.loglog(freq_, TX, "--", label="TianQin X")
    plt.loglog(freq_, TXY, "-.", label="TianQin XY")

    plt.xlabel("frequency")
    plt.ylabel("PSD [Hz^{-1}]")

    plt.legend(loc="best")

    plt.show()
