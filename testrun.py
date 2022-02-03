#!/usr/bin/env python3

from Integrate_Nu_Sol_Prime import *
from Integrated_Nu_Sol_Graphics import *
from generate_potential import *

#-------6-------#
# For reference:
# def numerov(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL=0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL=0.0, Analytic=False, UserFunction="", Overwrite=False, N_EVAL = 1, MASS=3678.21, HBAR = 315775.326864, Generate = True):

# def generate(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL = 0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL = 0.0, Analytic = False, UserFunction = "", Overwrite = False):

#generate("matrixtesting3D", GridInfo(3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0), Overwrite=True)
#generate("matrixtesting3D", GridInfo.load("generateinfomatrixtesting3D3D.dat"), Overwrite=True)


#numerov("matrixtesting3D", GridInfo(3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0), N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting2D", 2, GridInfo(XMIN=-1.0, XMAX=1.0, XDIV=20, YMIN=-1.0, YMAX=1.0, YDIV=20, ZLEVEL=4.32), N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting1D", 1, GridInfo(XLEVEL=0.0, YLEVEL=0.0, ZMIN=3.32, ZMAX=5.32, ZDIV=30), N_EVAL = 3, Overwrite=True)

#generate("2GraphicsTesting", GridInfo(3, -1.0, 1.0, 20, 0.0, -1.0, 1.0, 20, 0.0, 3.32, 5.32, 20, 0.0, False, False), Overwrite = True)
#Contour("2GraphicsTesting", 4.32)
# PotentialZGraphics("2GraphicsTesting", 1.0, 1.0)

#Contour("firstOrderNew", 0.0)

#numerov("LJBig", GridInfo(3, -2.0, 2.0, 40, 0.0, -2.0, 2.0, 40, 0.0, 2.32, 6.32, 40, 0.0), N_EVAL = 7, IgnoreM = True)

#Contour("LJBig", 4.32)

#Accuracy testing
def accuracyTesting():
	grid = GridInfo(3, -1.0, 1.0, 20, 0.0, -1.0, 1.0, 20, 0.0, 3.32, 5.32, 20, 0.0)

	allEvals = np.ndarray([])

	allEvals = numerov("LJnew", grid, N_EVAL = 7, IgnoreM = True)
	# allEvals = np.append(allEvals, numerov("LJold", grid, N_EVAL = 7, IgnoreM = False))

	grid.Analytic = True
	grid.ZMIN = -1.0
	grid.ZMAX = 1.0
	grid.UserFunction = "437.69 * (x**2 + y**2 + z**2)"
	allEvals = np.append(allEvals, numerov("firstOrderNew", grid, N_EVAL = 7, IgnoreM = True))
	# allEvals = np.append(allEvals, numerov("firstOrderOld", grid, N_EVAL = 7, IgnoreM = False))

	grid.UserFunction = "437.69 * (x**2 + y**2 + z**2) + 0.5 * (x**4 + y**4 + z**4)"
	allEvals = np.append(allEvals, numerov("secondOrderNew", grid, N_EVAL = 7, IgnoreM = True))
	# allEvals = np.append(allEvals, numerov("secondOrderOld", grid, N_EVAL = 7, IgnoreM = False))

	reshapedEvals = np.reshape(np.real(allEvals), (7, -1), 'F')

	np.savetxt("accuracytesting.csv", reshapedEvals, delimiter=",")
#accuracyTesting()