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
def PotentialGraph2D(Type, ProjectName, ZLEVEL):

	g = gp.GridInfo.load(ProjectName, 3)

	g.NDIM = 2
	g.ZLEVEL = ZLEVEL

	name = f"Potential for {ProjectName}"

	def getV(newZ):
		print("Generating for", newZ)
		g.ZLEVEL = newZ
		return gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)

	Graph2D(Type, name, g, ZLEVEL, getV)


def PsiGraph2D(Type, ProjectName, ZLEVEL, EVAL_NUM=0):
	g = gp.GridInfo.load(ProjectName, 3)
	evec = np.load("vecarray%s%sD.npy" %(ProjectName, g.NDIM))[EVAL_NUM]

	valFile = open("valout%s%sD.dat" %(ProjectName, g.NDIM))
	lines = valFile.readlines()

	name = f"Psi for {ProjectName}, Eval {EVAL_NUM} which is {lines[EVAL_NUM]}"

	def getSlice(newZ):
		_, _, hz = g.hxyz()
		div = round((newZ - g.ZMIN) / hz)
		return evec[:, :, div]

	Graph2D(Type, name, g, ZLEVEL, getSlice)


# g is a GridInfo defining the grid to graph
# getSlice(ZLEVEL) returns the grid at the given Z-level (Potential or psi)
def Graph2D(Type, name, g, ZLEVEL, getSlice):
	if type(ZLEVEL) != float:
		print("ZLEVEL is not in float format.")
	
	if Type != 'contour' and Type != 'heat' and Type != 'surface':
		print('type must be "contour", "heat", or "surface"')
	else: 

		xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		
		meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")

		if Type == "surface":
			fig = plt.figure(num=name)
			axis = fig.add_subplot(111, projection='3d')
		else:
			fig, axis = plt.subplots(num=name)

		zSlider = widgets.Slider(
			ax=plt.axes([0.25, 0.0, 0.65, 0.03]), 
			label='Z-level',
			valmin=g.ZMIN,
			valmax=g.ZMAX,
			valinit=ZLEVEL
		)

		def update(val):
			g.ZLEVEL = val
			
			newGrid = getSlice(val)

			axis.clear()

			if Type == 'contour':
				plotHandle = axis.contour(meshx, meshy, newGrid)
				axis.clabel(plotHandle, fontsize = 6)
			elif Type == 'heat':
				axis.pcolormesh(meshx, meshy, newGrid)
				axis.set_aspect('equal')
			elif Type == 'surface':
				axis.plot_surface(meshx, meshy, newGrid)

			plt.draw()

		# Draw first plot
		update(ZLEVEL)

		zSlider.on_changed(update)

		# Commented out to enable multiple graphs in the same run
		#plt.show()

		# We need to keep a global reference to the slider, because pyplot is stupid
		Graph2D.handles.append(zSlider)

Graph2D.handles = []

		
def PotentialZGraphics(ProjectName, XLEVEL, YLEVEL):
	if type(YLEVEL) != float:
		print("YLEVEL is not in float format.")
	elif type(XLEVEL) != float:
		print("XLEVEL is not in float format.")
	else:
		ZGraphicalGenerate(ProjectName, XLEVEL, YLEVEL)
		plt.plot(zgrid, V)
		plt.show()