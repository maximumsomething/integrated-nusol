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
from inus_common_util import *



#Integrated_Nu_Sol_Graphics has all the code for creating visualizations of what is going within the MOF. The code has the ability to provide graphs of 1D levels of the Potential and Eigenvectors along any specified axis
#as well as graphics for the Potential and Eigenvectors in the x-y plane with an adjustable Z-Level slider. 
#The code also has the ability to generate 2D level curves or "voxels"
#Finally, the code can also generate second derivative graphs for the potential's minimum to help with window sizing and spacing.

#A function to load the GridInfo for a specific ProjectName as well as generate a 1D potential along a specified axis at a specific 2D Level. 
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

	#depending on the axis specified generate the grid based on GridInfo 
	if axis == "x":
		axisgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
	if axis == "y":
		axisgrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
	if axis == "z":
		axisgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)

	return V, axisgrid, g


# Need to keep global reference to graphs, because pyplot is stupid
graphHandles = []

	

#A function to graph the potential in the x-y plane at a specified Z-Level.
# type is "contour", "heat", or "surface"
def PotentialGraph2D(Type, ProjectName, ZLEVEL):

	g = gp.GridInfo.load(ProjectName, 3)

	g.NDIM = 2
	g.ZLEVEL = ZLEVEL

	name = f"Potential for {ProjectName}"

	#Function to regenerate the potential when the slider is moved to a new Z-Level.
	def getV(newZ):
		print("Generating for", newZ)
		g.ZLEVEL = newZ
		return gp.generate(ProjectName, g, Overwrite = True, PrintAnalysis = False)

	Graph2D(Type, name, g, ZLEVEL, getV)

#Function to graph the eigenvectors of a specific Eigenvalue in the x-y plane at a specific ZLEVEL
def PsiGraph2D(Type, ProjectName, ZLEVEL, EVAL_NUM=0):
	g = gp.GridInfo.load(ProjectName, 3)
	evec = np.load(Filenames.vecarray(ProjectName, g.NDIM))[EVAL_NUM]

	#reads the eigenvalue file to find the specific eigenvalue
	valFile = open(Filenames.valout(ProjectName, g.NDIM))
	lines = valFile.readlines()

	name = f"Psi for {ProjectName}, Eval {EVAL_NUM} which is {lines[EVAL_NUM]}"

	#Function with a similar concept to getV(). This refreshed the graph whenever the Z-Level slider is moved. However, there are only discrete values for Z as the eigenvectors cannot be regenerated.
	#The function rounds to the nearest zdivision it has in the total eigenvector array. 
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
		#creates the grid of all points on the x-y plane
		meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")

		if Type == "surface":
			fig = plt.figure(num=name)
			axis = fig.add_subplot(111, projection='3d')
		else:
			fig, axis = plt.subplots(num=name)
		#Slider for the Zlevel for 2D graphs through the use of widgets

		zSlider = widgets.Slider(
			ax=plt.axes([0.25, 0.0, 0.65, 0.03]), 
			label='Z-level',
			valmin=g.ZMIN,
			valmax=g.ZMAX,
			valinit=ZLEVEL
		)
		#function that updates the Zlevel every time the slider is moved
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
	V = np.load(Filenames.potarray(ProjectName, 3))
	title = f"Potential for {ProjectName}"
	Voxel3D(ProjectName, title, V, level, minlev, maxlev)


def PsiVoxel3D(ProjectName, EVAL_NUM=0, level=None, minlev=None, maxlev=None):
	evec = np.load(Filenames.vecarray(ProjectName, 3))[EVAL_NUM]

	evec = np.square(evec)

	valFile = open(Filenames.valout(ProjectName, 3))
	lines = valFile.readlines()

	title = f"Psi^2 for {ProjectName}, Eval {EVAL_NUM} which is {lines[EVAL_NUM]}"
	Voxel3D(ProjectName, title, evec, level, minlev, maxlev)



# Don't call directly, use the PotentialVoxel3D and PsiVoxel3D functions instead
def Voxel3D(ProjectName, graphTitle, arr, level=None, minlev=None, maxlev=None):

	g = gp.GridInfo.load(ProjectName, 3)

	#if the minlevel or max level is not specified, it is loaded from the Array
	if minlev == None: minlev = np.amin(arr)
	if maxlev == None: maxlev = np.amax(arr)

	if level == None: level = np.average(arr)

	fig = plt.figure(num=graphTitle)
	axis = fig.add_subplot(111, projection='3d')

	#slider for Voxels
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



#function that graphs the Potential with respect to some axis on a certain 2D level		
def PotentialGraphics1D(ProjectName, XLEVEL = None, YLEVEL = None, ZLEVEL = None, axis = "z"):  

	V, axisgrid, _ = GraphicalGenerate1D(ProjectName, XLEVEL, YLEVEL, ZLEVEL, axis)
	plt.plot(axisgrid, V)


#function that graphs the Psi values with respect to some axis on a certain 2D level
def PsiGraphics1D(ProjectName, XLEVEL = None, YLEVEL = None, ZLEVEL = None, axis = "z", EVAL_NUM=0):
	g = gp.GridInfo.load(ProjectName, 3)
	evec = np.load(Filenames.vecarray(ProjectName, g.NDIM))[EVAL_NUM]



	hx, hy, hz = g.hxyz()
	divx = round((XLEVEL - g.XMIN) / hx)
	divy = round((YLEVEL - g.YMIN) / hy)
	divz = round((ZLEVEL - g.ZMIN) / hz)

	if axis == "x":
		#takes the specified level or slice from the eigenvector array
		slice = evec[:, divy, divz]
		axisgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
	elif axis == "y":
		slice = evec[divx, :, divz]
		axisgrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
	elif axis == "z":
		slice = evec[divx, divy, :]
		axisgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)
	else:
		raise ValueError("Axis must be x, y, or z")

	plt.plot(axisgrid, slice)
	plt.show()



#graphs the 2nd Derivatives of a potential in a specific axis
def Potential2ndDerGraph1D(ProjectName, XLEVEL = None, YLEVEL = None, ZLEVEL = None, axis = "z"):
	V, axisgrid, g = GraphicalGenerate1D(ProjectName, XLEVEL, YLEVEL, ZLEVEL, axis)

	hx, hy, hz = g.hxyz()

	if axis == "x": space = hx
	if axis == "y": space = hy
	if axis == "z": space = hz

	#difference quotient
	der = (V[0:-2] - 2*V[1:-1] + V[2:]) / space ** 2
	dergrid = axisgrid[1:-1]

	plt.plot(dergrid, der)





