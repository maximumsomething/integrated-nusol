import numpy as np
import sys
import operator
import os
import os.path
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
np.set_printoptions(threshold=sys.maxsize)

import generate_potential as gp

def ZGraphicalGenerate(ProjectName, XLEVEL, YLEVEL):
	g = gp.GridInfo.load(ProjectName, 3)
	g.NDIM = 1
	g.XLEVEL = XLEVEL
	g.YLEVEL = YLEVEL

	global V, zgrid
	V = gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)
	zgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)


def Contour(ProjectName, ZLEVEL, MINLEV=-.15, MAXLEV=.08, DIV=25):
	if type(ZLEVEL) != float:
		print("ZLEVEL is not in float format.")
	elif type(MINLEV) != float:
		print("MINLEV is not in float format.")
	elif type(MAXLEV) != float:
		print("MAXLEV is not in float format.")
	elif type(DIV) != int:
		print("DIV is not in an integer format. Make sure there are no decimals.")
	elif DIV <=0:
		print("DIV cannot be less than or equal to zero.")
	else: 
		g = gp.GridInfo.load(ProjectName, 3)
		g.NDIM = 2
		g.ZLEVEL = ZLEVEL

		xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		
		meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")

		fig, contour_axis = plt.subplots()

		zSlider = widgets.Slider(
			ax=plt.axes([0.25, 0.0, 0.65, 0.03]), 
			label='Z-level',
			valmin=g.ZMIN,
			valmax=g.ZMAX,
			valinit=ZLEVEL
		)

		def update(val):
			print("Generating for", val)
			g.ZLEVEL = val
			newGrid = gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)

			contour_axis.clear()

			CS1 = contour_axis.contour(meshx, meshy, newGrid)
			contour_axis.clabel(CS1, fontsize = 6)

			plt.draw()

		# Draw first plot
		update(ZLEVEL)

		zSlider.on_changed(update)

		plt.show()




def Heat(ProjectName, ZLEVEL):
	if type(ZLEVEL) != float:
		print("ZLEVEL is not in float format.")
	else:
		
		GraphicalGenerate(ProjectName, ZLEVEL)

		fig,ax = plt.subplots()
		ax.pcolormesh(meshx, meshy, V)
		ax.set_aspect('equal')
		plt.show()


def Surface(ProjectName, ZLEVEL):
	if type(ZLEVEL) != float:
		print("ZLEVEL is not in float format.")
	else:
		
		GraphicalGenerate(ProjectName, ZLEVEL)

		figsur = plt.figure()
		axsur = figsur.add_subplot(111, projection='3d')
		axsur.plot_surface(meshx, meshy, V)
		plt.show()
		
def PotentialZGraphics(ProjectName, XLEVEL, YLEVEL):
	if type(YLEVEL) != float:
		print("YLEVEL is not in float format.")
	elif type(XLEVEL) != float:
		print("XLEVEL is not in float format.")
	else:
		ZGraphicalGenerate(ProjectName, XLEVEL, YLEVEL)
		plt.plot(zgrid, V)
		plt.show()