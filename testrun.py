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


#numerov("matrixtesting3D", 3, -1.0, 1.0, 10, 0.0, -1.0, 1.0, 10, 0.0, 3.32, 5.32, 10, 0.0, N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting2D", 2, -1.0, 1.0, 20, 0.0, -1.0, 1.0, 20, 0.0, 0.0, 0.0, 0, 0.0, N_EVAL = 3, Overwrite=True)
#numerov("matrixtesting1D", 1, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 3.32, 5.32, 30, 0.0, N_EVAL = 3, Overwrite=True)

#generate("2GraphicsTesting", GridInfo(3, -1.0, 1.0, 15, 0.0, -1.0, 1.0, 15, 0.0, 3.32, 5.32, 15, 0.0, False, False), Overwrite = True, PrintAnalysis = True)
PotentialZGraphics("2GraphicsTesting", 1.0, 1.0)
#def test(grids):
	#start = time.process_time()
	#numerov("matrixtesting3D", 3, -1.0, 1.0, grids, 0.0, -1.0, 1.0, grids, 0.0, 3.32, 5.32, grids, 0.0, N_EVAL = 3, Overwrite=True)
	#print('total time for ', grids, ':', time.process_time() - start)


#test(10)
#test(15)
#test(21)
#test(27)
#test(35)
#test(42)
#test(53)
#test(67)
#test(74)
#test(86)
#test(100)
