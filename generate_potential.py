#-------1-------#
import numpy as np
import sys
import operator
import os
import os.path
from scipy.linalg import solve
import scipy.optimize as op
import scipy.sparse as sp
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from inus_common_util import *

#-------2-------#
class atom:
	def __init__ (self, x, y, z, charge=0, sigma=0, epsilon=0, mass=0):
		self.x=x
		self.y=y
		self.z=z
		self.charge=charge
		self.sigma=sigma
		self.epsilon=epsilon
		self.mass=mass

atoms = [atom(-1.855325180842072, 0.0, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
		 atom(0.9276625904210358, -1.606758738890191, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
		 atom(0.9276625904210358, 1.6067587388901914, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
		 atom(0.0, 0.0, 0.0, -2.2568, 3.118, 0.0000956054, 0),
		 atom(-2.2071535575638297, -1.575575329839865, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(2.468065039999284, -1.1236633859835425, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(-0.26091148243545437, 2.6992387158234075, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(2.468065039999284, 1.1236633859835425, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(-0.2609114824354546, -2.6992387158234075, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(-2.2071535575638292, 1.575575329839865, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
		 atom(-1.46152887986063, -2.53144227664784, 2.0669139636988607, 1.0982, 3.431, 0.000167333, 0),
		 atom(2.92305775972126, 0.0, 2.0669139636988607, -0.1378, 3.431, 0.000167333, 0),
		 atom(-1.4615288798606296, 2.53144227664784, 2.0669139636988607, -0.0518, 3.431, 0.000167333, 0)]

#------3-------#
hydrogensigma = 2.571
hydrogenepsilon = 0.0000701127
Ck = 8.9875517923E9
alpha = 1

#-------4-------#
def generate(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL = 0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL = 0.0, Analytic = False, UserFunction = "", Overwrite = False, PrintAnalysis = True):
	#-------4.1-------#
	if type(ProjectName) != str:
		print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
	elif type(NDIM) != int or NDIM>3 or NDIM<1:
		print("Number of diminsions must be in an integer format and be no greater than 3 and no less than 1.")
	elif (NDIM == 2 or NDIM == 3) and (type(XMIN)!= float or type(YMIN)!= float or type(XMAX)!= float or type(YMAX)!= float or type(XDIV)!= int or type(YDIV)!= int or XMIN>=XMAX or YMIN>=YMAX or XDIV<= 0 or YDIV<=0):
		print("XMIN, XMAX, YMIN, YMAX must all be floats and subject XMIN=<XMAX and YMIN=<YMAX. XDIV, YDIV must be integers greater than zero.")
	elif (NDIM == 1 or NDIM ==3) and (type(ZMIN)!= float or type(ZMAX)!= float or type(ZDIV)!= int or ZMIN>=ZMAX or ZDIV<=0):
		print("ZMIN, ZMAX must be floats and subject to ZMIN=<ZMAX. ZDIV must be an integer greater than zero.")
	elif NDIM == 1 and (type(XLEVEL)!= float or type(YLEVEL)!= float):
		print("XLEVEL, YLEVEL must be floats.")
	elif NDIM ==2 and type(ZLEVEL)!= float:
		print("ZLEVEL must be a float.")
	elif type(Analytic) != bool:
		print("Analytic is not in a boolean format. Make sure it is either true or false.")
	elif type(UserFunction) != str and Analytic == True:
			print("Function is not in a string format. Make sure the function is in quotation marks and contains only approproiate characters.")
	elif type(Overwrite) != bool:
		print("Overwrite is not in a boolean format. Make sure it is either true or false.")
		sys.exit()
	#-------4.2-------#
	else:
		print("Generating Potential...")
		startTimer("generate")
		LJPOL = np.array([])
		PotentialArrayPath = "Potential%s%sD.npy" %(ProjectName, NDIM)
		GenerateInfofile = "generateinfo%s%sD.dat" %(ProjectName, NDIM)

		checkSuccess = checkFileWriteable(PotentialArrayPath, "Potential Array", Overwrite)
		checkSuccess &= checkFileWriteable(GenerateInfofile, "Generate Info", Overwrite)

		if not checkSuccess:
			sys.exit()
				
	#-------4.3-------#
		if NDIM == 1:
			XMIN = XLEVEL
			XMAX = XLEVEL
			YMIN = YLEVEL
			YMAX = YLEVEL
		if NDIM == 2:
			ZMIN = ZLEVEL
			ZMAX = ZLEVEL


		if NDIM == 3 or NDIM == 2:
			hx = (XMAX - XMIN) / (XDIV - 1)
			hy = (YMAX - YMIN) / (YDIV - 1)
		else: 
			hx = 0.0
			hy = 0.0

		if NDIM == 3 or NDIM == 1:
			hz = (ZMAX - ZMIN) / (ZDIV - 1)
		else:
			hz = 0.0

		if NDIM == 2 and hx != hy:
			print("WARNING: hx and hy must be equal for NuSol to work, but instead, hx=",hx," and hy=",hy, sep="")
		elif NDIM == 3 and (hx != hy or hx != hz):
			print("WARNING: hx, hy, and hz must be equal for NuSol to work, but instead, hx=",hx,", hy=",hy," and hz=",hz, sep="")


		if Analytic == False:
			if NDIM == 1:
				V = np.zeros(ZDIV)
				for zcoord in range(0, ZDIV):
					zval = ZMIN + zcoord * hz
					pot = pointPotential(XLEVEL, YLEVEL, zval)
					V[zcoord] = pot

			elif NDIM == 2:
				V = np.zeros((XDIV, YDIV))
				for xcoord in range(0, XDIV):
					for ycoord in range(0, YDIV):
						xval = XMIN + xcoord * hx
						yval = YMIN + ycoord * hy
						pot = pointPotential(xval, yval, ZLEVEL)
						V[xcoord, ycoord] = pot
			
			elif NDIM == 3:
				V = np.zeros((XDIV, YDIV, ZDIV))
				for xcoord in range(0, XDIV):
					for ycoord in range(0, YDIV):
						for zcoord in range(0, ZDIV):
							xval = XMIN + xcoord * hx
							yval = YMIN + ycoord * hy
							zval = ZMIN + zcoord * hz
							pot = pointPotential(xval, yval, zval)
							V[xcoord, ycoord, zcoord] = pot
				
	#-------4.4-------#  
	  ###check if axis right###          
		elif Analytic == True:
			if NDIM == 1:
				try:
					Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
					z = Zgrid
					V = np.array(eval(UserFunction))
					hz = Zgrid[1] - Zgrid[0]
				except NameError:
					print("Invalid function. Make sure your function is a function of z and that all non-elementary operations are preceded by 'np.'")
					sys.exit()
			if NDIM == 2:
				try:
					Xgrid = np.linspace(XMIN, XMAX, XDIV)
					hx = Xgrid[1] - Xgrid[0]
					Ygrid = np.linspace(YMIN, YMAX, YDIV)
					hy = Ygrid[1] - Ygrid[0]
					x,y = np.meshgrid(Xgrid,Ygrid)
					print(UserFunction)
					V = np.array(eval(UserFunction))
				except NameError:
					print("Invalid function. Make sure your function is a function of x and y and that all non-elementary operations are proceded by 'np.'")
					sys.exit()
			if NDIM == 3:
				try:
					Xgrid = np.linspace(XMIN, XMAX, XDIV)
					hx = Xgrid[1] - Xgrid[0]
					Ygrid = np.linspace(YMIN, YMAX, YDIV)
					hy = Ygrid[1] - Ygrid[0]
					Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
					hz = Zgrid[1]-Zgrid[0]
					x,y,z = np.meshgrid(Xgrid, Ygrid, Zgrid)
					V = np.array(eval(UserFunction))
				except NameError:
					print("Invalid function. Make sure your function is a function of x and y and that all non-elementary operations are proceded by 'np.'")
					sys.exit()
					
	#-------4.5-------# 

		print("########################### \n Done generating potential! \n###########################)")
		#print(V)
		
		if PrintAnalysis == True:
			if np.isnan(np.sum(V)) == False and np.isinf(np.sum(V)) == False:
				print("Maximum potential:", np.amax(V), "\nMinimum potential:", np.amin(V), "\nMinimum potential's array position", np.unravel_index(np.argmin(V, axis=None), V.shape))
			
				#result = (np.where(V == np.amin(V)))
				result = np.unravel_index(np.argmin(V), np.shape(V))
				
		#-------4.6-------#
				
				if NDIM == 1:
					#listofcoordinates = list(zip(ZMAX-result[0]*hz))
					#for coord in listofcoordinates:
					#    min_list.append(coord)

					#print("The z position of the minimum is", (min_list))
					minimumpot = np.amin(V)
				
				
					zresult = result[0]
					try:
						zsecondderivative = ((V[zresult+1] - 2*minimumpot + V[zresult-1])/(hz**2))
						print("The second partial derivative with respect to z is", zsecondderivative)
					except:
						print("Undefined second partial derivative with respect to z.")
						zsecondderivative = float("Nan")
					ysecondderivative = float("Nan")
					xsecondderivative = float("Nan")
				if NDIM == 2:
					#listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy))
				
					#for coord in listofcoordinates:
					#    min_list.append(coord)

					#print("The x,y position of the minimum is", (min_list))


					minimumpot = np.amin(V)

					xresult = result[0]
					yresult = result[1]

					try:
						xsecondderivative = ((V[xresult+1, yresult] - 2*minimumpot + V[xresult-1, yresult])/(hx**2))
						print("The second partial derivative with respect to x is", xsecondderivative)
					except:
						print("Undefined second partial derivative with respect to x.")
						xsecondderivative = float("Nan")
					try:
						ysecondderivative = ((V[xresult, yresult+1] - 2*minimumpot + V[xresult, yresult-1])/(hy**2))
						print("The second partial derivative with respect to y is", ysecondderivative)
					except:
						print("Undefined second partial derivative with respect to y.")
						ysecondderivative = float("Nan")
					try: 
						print("Del Squared is", xsecondderivative+ysecondderivative)
						delsquared = xsecondderivative+ysecondderivative
					except:
						print("Del Squared is undefined.")
						delsquared = float("Nan")
					zsecondderivative = float("Nan")
				if NDIM == 3:
					#listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy, ZMAX-result[2]*hz))
					#for coord in listofcoordinates:
					#    min_list.append(coord)
					#print("The x,y,z position of the minimum is", (min_list))
					minimumpot = np.amin(V)
					xresult = result[0]
					yresult = result[1]
					zresult = result[2]
					print(xresult, yresult, zresult)
					try:
						xsecondderivative = ((V[xresult+1, yresult, zresult] - 2*minimumpot + V[xresult-1, yresult, zresult])/(hx**2))
						print("The second partial derivative with respect to x is", xsecondderivative)
					except:
						print("Undefined second partial derivative with respect to x.")
						xsecondderivative = float("nan")
					try:
						ysecondderivative = ((V[xresult, yresult+1, zresult] - 2*minimumpot + V[xresult, yresult-1, zresult])/(hy**2))
						print("The second partial derivative with respect to y is", ysecondderivative)
					except:
						print("Undefined second partial derivative with respect to y.")
						ysecondderivative = float("nan")
					try:
						zsecondderivative = ((V[xresult, yresult, zresult+1] - 2*minimumpot + V[xresult, yresult, zresult-1])/(hz**2))
						print("The second partial derivative with respect to z is", zsecondderivative)
					except:
						print("Undefined second partial derivative with respect to z.")
						zsecondderivative = float("nan")
					try: 
						print("Del Squared is", xsecondderivative+ysecondderivative+zsecondderivative)
						delsquared = xsecondderivative+ysecondderivative+zsecondderivative
					except:
						print("Del Squared is undefined.")
						delsquared = float("nan")

		#-------4.7-------#
			
			print("########################### \nSaving the potential array as", PotentialArrayPath, "\n###########################")
			np.save(PotentialArrayPath, V)
			try:
				f = open(GenerateInfofile, 'w')
				if np.isnan(np.sum(V)) == False and np.isinf(np.sum(V)) == False:
					print("ProjectName = %s NDIM = %d XMIN = %.8f XMAX = %.8f XDIV = %d YMIN = %.8f YMAX = %.8f YDIV = %d ZMIN = %.8f ZMAX = %.8f ZDIV = %d Analytic = %s UserFunction = %s Overwrite = %s MAXPOT = %.8f MINPOT = %.8f XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f ZSECONDDERIVATIVE = %.8f" % (ProjectName,NDIM,XMIN,XMAX,XDIV,YMIN,YMAX,YDIV,ZMIN,ZMAX,ZDIV,Analytic,UserFunction,Overwrite, np.amax(V), np.amin(V), xsecondderivative, ysecondderivative, zsecondderivative), file=f)
				elif np.isnan(np.sum(V)) == True or np.isinf(np.sum(V)) == True:
					print("ProjectName = %s NDIM = %d XMIN = %.8f XMAX = %.8f XDIV = %d YMIN = %.8f YMAX = %.8f YDIV = %d ZMIN = %.8f ZMAX = %.8f ZDIV = %d Analytic = %s UserFunction = %s Overwrite = %s MAXPOT = DNE MINPOT = DNE XSECONDDERIVATIVE = DNE YSECONDDERIVATIVE = DNE ZSECONDDERIVATIVE = DNE" % (ProjectName,NDIM,XMIN,XMAX,XDIV,YMIN,YMAX,YDIV,ZMIN,ZMAX,ZDIV,Analytic,UserFunction,Overwrite), file=f)
				f.close()
			except IOError:
				print("Error: The potential did not save. The file you wanted to save to was already opened. Close the file and rerun the program.")
				sys.exit()
		
		return V


def pointPotential(xval, yval, zval):
	LJ = 0
	for atom in atoms:
		jointsigma = (atom.sigma + hydrogensigma)/2
		jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
		magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(zval-atom.z)**2)
		LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
		LJ += LJpointval

	return LJ
