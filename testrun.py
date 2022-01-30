#!/usr/bin/env python2

from Integrate_Nu_Sol_Prime import *


#-------6-------#
#generate("splittest", 1, 0.0, 0.0, 0, 13.0, 0.0, 0.0, 0, 13.0, -4.0, 4.0, 15, Overwrite = True)
numerov("matrixtesting3D", 3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0, N_EVAL = 3, Overwrite=True)