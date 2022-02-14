import numpy as np
import sys
import operator
import os
import os.path
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

# matplotlib.use('module://mplopengl.backend_qtgl')

from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.set_printoptions(threshold=sys.maxsize)

import generate_potential as gp



def GraphicalGenerate1D(ProjectName, XLEVEL, YLEVEL, ZLEVEL, axis):
	g = gp.GridInfo.load(ProjectName, 3)
	g.NDIM = 1
	# if levels are unspecified, use middle of window
	if XLEVEL == None: XLEVEL = (g.XMIN + g.XMAX) / 2
	if YLEVEL == None: YLEVEL = (g.YMIN + g.YMAX) / 2
	if ZLEVEL == None: ZLEVEL = (g.ZMIN + g.ZMAX) / 2

	if type(YLEVEL) != float:
		raise ValueError("YLEVEL is not in float format.")
	elif type(XLEVEL) != float:
		raise ValueError("XLEVEL is not in float format.")

	g.XLEVEL = XLEVEL
	g.YLEVEL = YLEVEL
	g.ZLEVEL = ZLEVEL
	g.axis = axis

	print(f"generating along axis {axis}; XLEVEL={XLEVEL} YLEVEL={YLEVEL} ZLEVEL={ZLEVEL}")

	V = gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)
	#V = V - (np.amin(V))
	#print(V)
	if axis == "x":
		axisgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
	if axis == "y":
		axisgrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
	if axis == "z":
		axisgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)

	return V, axisgrid


# Need to keep global reference to graphs, because pyplot is stupid
graphHandles = []

	

#def ZGraphicalGenerate(ProjectName, XLEVEL, YLEVEL):
#	g = gp.GridInfo.load(ProjectName, 3)
#	g.NDIM = 1
#	g.XLEVEL = XLEVEL
#	g.YLEVEL = YLEVEL

#	global V, zgrid
#	V = gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)
#	V = V - (np.amin(V))
#	zgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)


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


# Don't call directly, use the PotentialGraph2D and PsiGraph2D functions instead
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

			plt.tight_layout()

			plt.draw()

		# Draw first plot
		update(ZLEVEL)

		zSlider.on_changed(update)

		# Commented out to enable multiple graphs in the same run
		#plt.show()

		# We need to keep a global reference to the slider, because pyplot is stupid
		graphHandles.append(zSlider)



def PotentialVoxel3D(ProjectName, level=0.0, minlev=None, maxlev=None):
	V = np.load("Potential%s3D.npy" % (ProjectName))
	title = f"Potential for {ProjectName}"
	Voxel3D(ProjectName, title, V, level, minlev, maxlev)


def PsiVoxel3D(ProjectName, EVAL_NUM=0, level=None, minlev=None, maxlev=None):
	evec = np.load("vecarray%s%sD.npy" %(ProjectName, 3))[EVAL_NUM]

	evec = np.square(evec)

	valFile = open("valout%s%sD.dat" %(ProjectName, 3))
	lines = valFile.readlines()

	title = f"Psi^2 for {ProjectName}, Eval {EVAL_NUM} which is {lines[EVAL_NUM]}"
	Voxel3D(ProjectName, title, evec, level, minlev, maxlev)



# Don't call directly, use the PotentialVoxel3D and PsiVoxel3D functions instead
def Voxel3D(ProjectName, graphTitle, arr, level=None, minlev=None, maxlev=None):

	g = gp.GridInfo.load(ProjectName, 3)

	if minlev == None: minlev = np.amin(arr)
	if maxlev == None: maxlev = np.amax(arr)

	if level == None: level = np.average(arr)

	fig = plt.figure(num=graphTitle)
	axis = fig.add_subplot(111, projection='3d')

	levelSlider = widgets.Slider(
		ax=plt.axes([0.25, 0.0, 0.65, 0.03]), 
		label='Boundary level',
		valmin=minlev,
		valmax=maxlev,
		valinit=level
	)
	def update(val):
		axis.clear()

		# https://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html

		verts, faces, normals, values = marching_cubes(arr, val, spacing=g.hxyz())
		verts += (g.XMIN, g.YMIN, g.ZMIN) # Correct axis
		mesh = Poly3DCollection(verts[faces])
		# mesh.set_edgecolor('k')

		faceNormals = np.average(normals[faces], axis=1)

		# rainbow
		# colors = (faceNormals + 1) / 2

		# grayscale
		lightSource = [0, 0, 1]
		colors = (np.dot(faceNormals, lightSource) + 1) / 2
		colors = np.vstack((colors, colors, colors)).T

		mesh.set_facecolors(colors)
		axis.add_collection3d(mesh)

		axis.set_xlim(g.XMIN, g.XMAX)
		axis.set_ylim(g.YMIN, g.YMAX)
		axis.set_zlim(g.ZMIN, g.ZMAX)

		plt.tight_layout()

		#axis.voxels(filled=(V < val))
		plt.draw()

	levelSlider.on_changed(update)
	update(level)

	graphHandles.append(levelSlider)



		
def PotentialGraphics1D(ProjectName, XLEVEL = None, YLEVEL = None, ZLEVEL = None, axis = "z"):  

	V, axisgrid = GraphicalGenerate1D(ProjectName, XLEVEL, YLEVEL, ZLEVEL, axis)
	plt.plot(axisgrid, V)
	plt.show()