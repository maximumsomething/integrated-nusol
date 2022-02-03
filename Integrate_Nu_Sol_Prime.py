import numpy as np
import subprocess
import operator
import os
import sys
from scipy.linalg import solve
from datetime import datetime
import scipy.optimize as op
import scipy.sparse as sp
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from generate_potential import generate
from inus_common_util import *


USE_FEAST=False

#-------5-------#       



# HBAR in terms of ((J^2*s^2)(K/J)(m_u/Kg)(A/m)^2)^ 1/2
# = (hbar)^2 / Kb / m_u * 10^20x`
# where hbar is planck's constant in joule-seconds, Kb is Boltzmann's constant, m_u is atomic mass in kilograms, and 10^20 is square angstroms per square meter.

def numerov(ProjectName, gridInfo, Overwrite=False, N_EVAL = 1, MASS=2.0, HBAR = 6.96482118, IgnoreM = True, Generate = True):
	g = gridInfo # for shortness
#-------5.1-------# 
	if type(Generate) != bool: 
		print("Generate is not in a boolean format. Make sure it is either true or false.")
		sys.exit()
	elif type(Overwrite) != bool:
		print("Overwrite is not in a boolean format. Make sure it is either true or false.")
		sys.exit()
	elif type(ProjectName) != str:
		print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
		sys.exit()
	elif type(N_EVAL) != int or N_EVAL<1:
		print("N_EVAL must be an integer greater than zero.")
		sys.exit()
	elif type(MASS) != float or MASS <= 0:
		print("MASS must be a float and be greater than zero.")
		sys.exit()
	elif type(HBAR) != float:
		print("HBAR must be a float.")
		sys.exit()


#-------5.2-------# 
	PotentialArrayPath = "Potential%s%sD.npy" %(ProjectName, g.NDIM)
	GenerateInfofile = "generateinfo%s%sD.dat" %(ProjectName, g.NDIM)
	EIGENVALUES_OUT = "valout%s%sD.dat" %(ProjectName, g.NDIM)
	EIGENVECTORS_OUT = "vecout%s%sD.dat" %(ProjectName, g.NDIM)
	Eigenvectoranalysis = "vecanalysis%s%sD.npy" %(ProjectName, g.NDIM)


	checkSuccess = checkFileWriteable(EIGENVALUES_OUT, "eigenvalue out", Overwrite)
	checkSuccess &= checkFileWriteable(EIGENVECTORS_OUT, "eigenvector out", Overwrite)
	checkSuccess &= checkFileWriteable(Eigenvectoranalysis, "eigenvector analysis", Overwrite)
	
	if Generate == False:
		checkSuccess &= checkFileReadable(PotentialArrayPath, "Potential Array")
		checkSuccess &= checkFileReadable(GenerateInfofile, "Generate Info")
	else:
		checkSuccess &= checkFileWriteable(PotentialArrayPath, "Potential Array", Overwrite)
		checkSuccess &= checkFileWriteable(GenerateInfofile, "Generate Info", Overwrite)

	if not checkSuccess:
		sys.exit()
		
#-------5.5-------#
	if Generate == True:
		V = generate(ProjectName, g, Overwrite)
#-------5.6-------# 
	else:
		V = np.load(PotentialArrayPath)


	startTimer("Create numerov matrices")

	hx, hy, hz = g.hxyz()

	if g.NDIM == 1:
		A, M = createNumerovMatrices1D(V, g.ZDIV, hz, MASS, HBAR)
	elif g.NDIM == 2:
		A, M = createNumerovMatrices2D(V, g.XDIV, g.YDIV, hx, MASS, HBAR)
	elif g.NDIM == 3:
		A, M = createNumerovMatrices3D(V, g.XDIV, g.YDIV, g.ZDIV, hx, MASS, HBAR)
	
	if USE_FEAST:
		runFeast(A, M, EIGENVALUES_OUT, EIGENVECTORS_OUT)
	else:
		startTimer("solve eigs")
		if IgnoreM:
			eval, evec = solveEigsApprox(A, N_EVAL)
		else:
			eval, evec = solveEigs(A, M, N_EVAL)
		endTimer()
		writeEigs(eval, evec, EIGENVALUES_OUT, EIGENVECTORS_OUT, Eigenvectoranalysis)

		return eval

#-------5.7-------# 
def createNumerovMatrices1D(V, ZDIV, hz, MASS, HBAR):
	preFactor1D = -6.0* HBAR * HBAR / (MASS * hz * hz)
	NumerovMatrix1D = []
	FORTRANoffset = 1
	Nele = 0
	for i in range(ZDIV):
		NumerovMatrix1D.append(
			[FORTRANoffset + i, FORTRANoffset + i, -2.0 * preFactor1D + 10.0 * V[i], 10.0])
		Nele += 1
		if i - 1 >= 0:
			NumerovMatrix1D.append(
				[FORTRANoffset + i, FORTRANoffset + i - 1, 1.0 * preFactor1D + V[i - 1], 1.0])
			Nele += 1
		if i + 1 < ZDIV:
			NumerovMatrix1D.append(
				[FORTRANoffset + i, FORTRANoffset + i + 1, 1.0 * preFactor1D + V[i + 1], 1.0])
			Nele += 1
	NumerovMatrix1D = sorted(NumerovMatrix1D, key=operator.itemgetter(0, 1))
	NumerovMatrix1D = np.array(NumerovMatrix1D)
	row = NumerovMatrix1D[:, 0] - 1
	col = NumerovMatrix1D[:, 1] - 1
	dataA = NumerovMatrix1D[:, 2]
	dataM = NumerovMatrix1D[:, 3]
	A = sp.coo_matrix((dataA, (row, col)), shape=(ZDIV, ZDIV))
	M = sp.csr_matrix((dataM, (row, col)), shape=(ZDIV, ZDIV))
	return (A, M)

#-------5.8-------#
def createNumerovMatrices2D(V, XDIV, YDIV, hx, MASS, HBAR):
	preFactor2D =  (HBAR * HBAR) / (MASS * hx * hx)
	Nx = XDIV
	Ny = YDIV
	NumerovMatrix2D = []
	FORTRANoffset = 1
	Nele = 0

	for iN in range(Nx):
		for iK in range(Ny):
			if (iN - 1 >= 0):
				iNx = iN * Ny
				iNy = (iN - 1) * Ny
				iKx = iK
				iKy = iK
				if (iKy - 1 >= 0):
					NumerovMatrix2D.append(
						[FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1, -   1.0 * preFactor2D, 0.0])
					Nele += 1
				NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
									  -   4.0 * preFactor2D + V[iN - 1, iK], 1.0])
				Nele += 1
				if (iKy + 1 < Ny):
					NumerovMatrix2D.append(
						[FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1, -   1.0 * preFactor2D, 0.0])
					Nele += 1

			iNx = iN * Ny
			iNy = iN * Ny
			iKx = iK
			iKy = iK
			if (iKy - 1 >= 0):
				NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1,
									  -  4.0 * preFactor2D + V[iN, iK - 1], 1.0])
				Nele += 1
			NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
									+ 20.0 * preFactor2D + 8.0 * V[iN, iK], 8.0])
			Nele += 1
			if (iKy + 1 < Ny):
				NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1,
									  -  4.0 * preFactor2D + V[iN, iK + 1], 1.0])
				Nele += 1

			if (iN + 1 < Nx):
				iNx = iN * Ny
				iNy = (iN + 1) * Ny
				iKx = iK
				iKy = iK
				if (iKy - 1 >= 0):
					NumerovMatrix2D.append(
							[FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1, -  1.0 * preFactor2D, 0.0])
					Nele += 1
					NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
											-  4.0 * preFactor2D + V[iN + 1, iK], 1.0])
					Nele += 1
					if (iKy + 1 < Ny):
						NumerovMatrix2D.append(
							[FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1, -  1.0 * preFactor2D, 0.0])
						Nele += 1


	NumerovMatrix2D = sorted(NumerovMatrix2D, key=operator.itemgetter(0, 1))

	NumerovMatrix2D = np.array(NumerovMatrix2D)
	row = NumerovMatrix2D[:, 0] - 1
	col = NumerovMatrix2D[:, 1] - 1
	dataA = NumerovMatrix2D[:, 2]
	dataM = NumerovMatrix2D[:, 3]
	A = sp.coo_matrix((dataA, (row, col)), shape=(XDIV * YDIV, XDIV * YDIV))
	M = sp.csr_matrix((dataM, (row, col)), shape=(XDIV * YDIV, XDIV * YDIV))
	return (A, M)

#-------5.9-------#
def createNumerovMatrices3D(V, XDIV, YDIV, ZDIV, hx, MASS, HBAR):
	preFactor3D = - (HBAR * HBAR) / (2.0 * MASS * hx * hx)
	Nx = XDIV
	Ny = YDIV
	Nz = ZDIV
	NumerovMatrix3D = []
	FORTRANoffset = 1
	Nele = 0
	for iL in range(Nz):
	  # process l-1 block
	  # NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy - 1 ,  1.0 * self.cfg.preFactor3D , 0.0 ) )
		if (iL - 1 >= 0):
			iLx = (iL) * Ny * Nx
			iLy = (iL - 1) * Ny * Nx
			for iN in range(Nx):
				for iK in range(Ny):
					if (iN - 1 >= 0):
						iNx = iN * Ny
						iNy = (iN - 1) * Ny
						iKx = iK
						iKy = iK

						if (iKy - 1 >= 0):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * preFactor3D, 0.0])
						Nele += 1
						if (iKy + 1 < Ny):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1

					iNx = iN * Ny
					iNy = iN * Ny
					iKx = iK
					iKy = iK
					if (iKy - 1 >= 0):
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * preFactor3D,
							 0.0])
						Nele += 1
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
									16.0 * preFactor3D + V[iN, iK, iL - 1], 1.0])
					Nele += 1
					if (iKy + 1 < Ny):
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * preFactor3D,
							 0.0])
						Nele += 1

					if (iN + 1 < Nx):
						iNx = iN * Ny
						iNy = (iN + 1) * Ny
						iKx = iK
						iKy = iK
						if (iKy - 1 >= 0):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * preFactor3D, 0.0])
						Nele += 1
						if (iKy + 1 < Ny):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1

	  # l
		iLx = (iL) * Ny * Nx
		iLy = (iL) * Ny * Nx
		for iN in range(Nx):
			for iK in range(Ny):
				if (iN - 1 >= 0):
					iNx = iN * Ny
					iNy = (iN - 1) * Ny
					iKx = iK
					iKy = iK
					if (iKy - 1 >= 0):
						NumerovMatrix3D.append(
						[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * preFactor3D,
						 0.0])
						Nele += 1
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
									16.0 * preFactor3D + V[iN - 1, iK, iL], 1.0])
					Nele += 1
					if (iKy + 1 < Ny):
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * preFactor3D,
							 0.0])
						Nele += 1

				iNx = iN * Ny
				iNy = iN * Ny
				iKx = iK
				iKy = iK
				if (iKy - 1 >= 0):
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1,
									16.0 * preFactor3D + V[iN, iK - 1, iL], 1.0])
					Nele += 1

				NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
								  -72.0 * preFactor3D + 6.0 * V[iN, iK, iL], +6.0])
				Nele += 1

				if (iKy + 1 < Ny):
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1,
									16.0 * preFactor3D + V[iN, iK + 1, iL], 1.0])
					Nele += 1

				if (iN + 1 < Nx):
					iNx = iN * Ny
					iNy = (iN + 1) * Ny
					iKx = iK
					iKy = iK
					if (iKy - 1 >= 0):
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * preFactor3D,
							 0.0])
						Nele += 1
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
									16.0 * preFactor3D + V[iN + 1, iK, iL], 1.0])
					Nele += 1
					if (iKy + 1 < Ny):
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * preFactor3D,
							0.0])
						Nele += 1

		if (iL + 1 < Nz):
			iLx = (iL) * Ny * Nx
			iLy = (iL + 1) * Ny * Nx
			for iN in range(Nx):
				for iK in range(Ny):
					if (iN - 1 >= 0):
						iNx = iN * Ny
						iNy = (iN - 1) * Ny
						iKx = iK
						iKy = iK
						if (iKy - 1 >= 0):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * preFactor3D, 0.0])
						Nele += 1
						if (iKy + 1 < Ny):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
					iNx = iN * Ny
					iNy = iN * Ny
					iKx = iK
					iKy = iK
					if (iKy - 1 >= 0):
						NumerovMatrix3D.append(
						[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * preFactor3D,
						 0.0])
						Nele += 1
					NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
									16.0 * preFactor3D + V[iN, iK, iL + 1], 1.0])
					Nele += 1
					if (iKy + 1 < Ny):
						NumerovMatrix3D.append(
						[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * preFactor3D,
						 0.0])
						Nele += 1
					if (iN + 1 < Nx):
						iNx = iN * Ny
						iNy = (iN + 1) * Ny
						iKx = iK
						iKy = iK
						if (iKy - 1 >= 0):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
						NumerovMatrix3D.append(
							[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * preFactor3D, 0.0])
						Nele += 1
						if (iKy + 1 < Ny):
							NumerovMatrix3D.append(
								[FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * preFactor3D,
								0.0])
							Nele += 1
			   
	NumerovMatrix3D = sorted(NumerovMatrix3D, key=operator.itemgetter(0, 1))
	NumerovMatrix3D = np.array(NumerovMatrix3D)
	row = NumerovMatrix3D[:, 0] - 1
	col = NumerovMatrix3D[:, 1] - 1
	dataA = NumerovMatrix3D[:, 2]
	dataM = NumerovMatrix3D[:, 3]
	A = sp.coo_matrix((dataA, (row, col)), shape=(
	XDIV * YDIV * ZDIV, XDIV * YDIV * ZDIV))
	M = sp.csr_matrix((dataM, (row, col)), shape=(
	XDIV * YDIV * ZDIV, ZDIV * YDIV * ZDIV))
	return (A, M)

#-------5.10-------# 

def solveEigsApprox(A, N_EVAL):
	eval, evec =  sp.linalg.eigs(A=A, k=N_EVAL, which='SM')
	eval /= 12
	return eval, evec

def solveEigs(A, M, N_EVAL):
	#startTimer("convert to dense")
	#MDense = M.todense()
	#startTimer("Invert M")
	#Minv = np.linalg.inv(MDense)
	#startTimer("Sparsify M")
	#Minv = sp.coo_matrix(Minv)

	#startTimer("Multiply Minv")
	#Q = A * Minv
	
	#eval, evec = sp.linalg.eigs(A=Q, k=N_EVAL, which='SM')

	return sp.linalg.eigs(A=A, k=N_EVAL, M=M, which='SM')

def writeEigs(eval, evec, EIGENVALUES_OUT, EIGENVECTORS_OUT, Eigenvectoranalysis):
	
	norder = eval.argsort()
	eval = eval[norder].real
	evec = evec.T[norder].real
	f = open(EIGENVALUES_OUT,'w')
	for e in eval:
		print("%.12f" % (e), file=f)
	f.close()

	f = open(EIGENVECTORS_OUT,'w')
	for e in evec:
		line=''
		for i in e:
			line+="%.12e " % i
		print(line, file=f)
	f.close()
	print("Save complete!")
	print("Saving Eigenvector Analysis File...")
	# Not sure if this was what was intended to be saved?
	np.save(Eigenvectoranalysis, evec)
	print("Eigenvector Analysis File Saved!")

def runFeast(A, M, EIGENVALUES_OUT, EIGENVECTORS_OUT):
	FEAST_COMMAND="wsl LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin/ $(wslpath 'C:\\Users\\Student\\Desktop\\NuSol_original\\lib\\NuSol_FEAST')"
	FEAST_MATRIX_OUT_PATH = "./FEAST_MATRIX"

	# This value must be larger than the expected number of eigenvalues in your search interval, and smaller than NGRIX*NGIDY*NGRIDZ. Otherwise it will not work!
	FEAST_M = 1000
	# Lower bound for eigenvalue solver search interval [Hartree]
	FEAST_E_MIN = 1000.0
	# Upper bound for eigenvalue solver search interval [Hartree]
	FEAST_E_MAX = 1000000000.0

	startTimer("write FEAST_MATRIX")
	f = open(FEAST_MATRIX_OUT_PATH,'w')

	(matsize, _) = A.get_shape()
	print("%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % \
		  (matsize, A.getnnz(), XDIV, YDIV, ZDIV, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, hz, hz, hz), file=f)
	# https://stackoverflow.com/a/4319159
	for row, col, data in zip(A.row, A.col, A.data):
		print("%12d%12d % 18.16E % 18.16E" % (row + 1, col + 1, data, M[row, col]), file=f)

	f.close()

	startTimer("call FEAST")

	p = subprocess.Popen('%s %f %f %d %s %s %s' % (FEAST_COMMAND,FEAST_E_MIN,FEAST_E_MAX,FEAST_M,FEAST_MATRIX_OUT_PATH,EIGENVALUES_OUT,EIGENVECTORS_OUT),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	for line in p.stdout.readlines():
		print(line, end="")
	retval = p.wait()

	endTimer()

