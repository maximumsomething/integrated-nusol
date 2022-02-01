#!/usr/bin/env python3

from Integrate_Nu_Sol_Prime import *
from generate_potential import *


#-------6-------#
# For reference:
# def numerov(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL=0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL=0.0, Analytic=False, UserFunction="", Overwrite=False, N_EVAL = 1, MASS=3678.21, HBAR = 315775.326864, Generate = True):

# def generate(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL = 0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL = 0.0, Analytic = False, UserFunction = "", Overwrite = False):

#generate("matrixtesting3D", GridInfo(3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0), Overwrite=True)
generate("matrixtesting3D", GridInfo.load("matrixtesting3D", 3), Overwrite=True)


#numerov("matrixtesting3D", GridInfo(3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0), N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting2D", 2, XMIN=-1.0, XMAX=1.0, XDIV=20, YMIN=-1.0, YMAX=1.0, YDIV=20, ZLEVEL=4.32, N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting1D", 1, XLEVEL=0.0, YLEVEL=0.0, ZMIN=3.32, ZMAX=5.32, ZDIV=30, N_EVAL = 3, Overwrite=True)