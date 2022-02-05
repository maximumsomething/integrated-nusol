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


# type is "contour", "heat", or "surface"
def Graph2D(Type, ProjectName, ZLEVEL):
	if type(ZLEVEL) != float:
		print("ZLEVEL is not in float format.")
	# elif type(MINLEV) != float:
	# 	print("MINLEV is not in float format.")
	# elif type(MAXLEV) != float:
	# 	print("MAXLEV is not in float format.")
	# elif type(DIV) != int:
	# 	print("DIV is not in an integer format. Make sure there are no decimals.")
	# elif DIV <=0:
	# 	print("DIV cannot be less than or equal to zero.")
	elif Type != 'contour' and Type != 'heat' and Type != 'surface':
		print('type must be "contour", "heat", or "surface"')
	else: 
		g = gp.GridInfo.load(ProjectName, 3)
		g.NDIM = 2
		g.ZLEVEL = ZLEVEL

		xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		
		meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")

		if Type == "surface":
			fig = plt.figure()
			axis = fig.add_subplot(111, projection='3d')
		else:
			fig, axis = plt.subplots()

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
			V = gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)

			axis.clear()

			if Type == 'contour':
				plotHandle = axis.contour(meshx, meshy, V)
				axis.clabel(plotHandle, fontsize = 6)
			elif Type == 'heat':
				axis.pcolormesh(meshx, meshy, V)
				axis.set_aspect('equal')
			elif Type == 'surface':
				axis.plot_surface(meshx, meshy, V)

			plt.draw()

		# Draw first plot
		update(ZLEVEL)

		zSlider.on_changed(update)

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