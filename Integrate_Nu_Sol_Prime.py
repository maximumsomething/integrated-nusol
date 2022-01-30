import numpy as np
import subprocess
import sys
import operator
import os
import os.path
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
#OVERWRITING#
def numerov(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL=0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL=0.0, Analytic=False, UserFunction="", Overwrite=False, N_EVAL = 1, MASS=3678.21, HBAR = 315775.326864, Generate = True):
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
		PotentialArrayPath = "Potential%s%sD.npy" %(ProjectName, NDIM)
		GenerateInfofile = "generateinfo%s%sD.dat" %(ProjectName, NDIM)
		EIGENVALUES_OUT = "valout%s%sD.dat" %(ProjectName, NDIM)
		EIGENVECTORS_OUT = "vecout%s%sD.dat" %(ProjectName, NDIM)
		Eigenvectoranalysis = "vecanalysis%s%sD.dat" %(ProjectName, NDIM)
		
		file_exists1 = os.path.exists(PotentialArrayPath)
		file_exists2 = os.path.exists(GenerateInfofile)
		file_exists3 = os.path.exists(EIGENVALUES_OUT)
		file_exists4 = os.path.exists(EIGENVECTORS_OUT)
		file_exists5 = os.path.exists(Eigenvectoranalysis)
	#-------5.3-------# 
		if Overwrite == True:
			try:
				myfile = open(PotentialArrayPath, "w")
				myfile.close()
				myfile = open(GenerateInfofile, "w")
				myfile.close()
				myfile = open(EIGENVALUES_OUT, "w")
				myfile.close()
				myfile = open(EIGENVECTORS_OUT, "w")
				myfile.close()
				myfile = open(Eigenvectoranalysis, "w")
				myfile.close()
			except IOError:
				print("The file(s) you are trying to save to are currently open. Close the file(s) and rerun the program again.")
				sys.exit()
	#-------5.4-------#    
		if Overwrite == False:
			if file_exists3 == True:
				print("The eigenvalue out file name already exists. Change the name or set overwrite to 'True'.")
				sys.exit()
			if file_exists4 == True:
				print("The eigenvector out file name already exists. Change the name or set overwrite to 'True'.")
				sys.exit()
			if file_exists5 == True:
				print("The eigenvector analysis out file name already exists. Change the name or set overwrite to 'True'.")
				sys.exit()
			if file_exists1 == True and Generate == True:
				print("The potential array path you are trying to save to already exists. Change its name or set overwrite to 'True'.")
				sys.exit()
			if file_exists2 == True and Generate == True:
				print("The generate info file you are trying to save to already exists. Change its name or set overwrite to 'True'.")
				sys.exit()
			if file_exists1 == False and Generate == False:
				print("The potential array file you are trying to access for NuSol does not exist.")
				sys.exit()
			if file_exists2 == False and Generate == False:
				print("The generate info file you are trying to acces for NuSol does not exist.")
				sys.exit()
			
	#-------5.5-------#
		if Generate == True:
			V = generate(ProjectName, NDIM, XMIN, XMAX, XDIV, XLEVEL, YMIN, YMAX, YDIV, YLEVEL, ZMIN, ZMAX, ZDIV, ZLEVEL, Analytic, UserFunction, Overwrite)
	#-------5.6-------# 
		hx = (XMAX - XMIN) / XDIV
		hz = (ZMAX - ZMIN) / ZDIV

			#fix
		if Generate == False:
			try:
				myfile = open(GenerateInfofile, 'w')
			except IOError:
				print("Please close the Generate Info File and re-run the program again.")
				sys.exit()
			lines = myfile.readlines()
			for line in lines:
				line = line.strip()
				line = line.split()
		   
			XDIV = float(line[14])
			YDIV = float(line[26])
			ZDIV = float(line[38])
			
			V= np.load(PotentialArrayPath)
	#-------5.7-------#        
		startTimer("Create numerov matrices")

		if NDIM == 1:
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
			return A, M
	#-------5.8-------#
		
		if NDIM == 2:
			preFactor2D =   1.0 / (MASS * hx * hx / (HBAR * HBAR))
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
			return A, M
	#-------5.9-------#
		
		if NDIM == 3:
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
			# return A, M
	#-------5.10-------# 
		#startTimer("convert to dense")
		#MDense = M.todense()
		#startTimer("Invert M")
		#Minv = np.linalg.inv(MDense)
		#startTimer("Sparsify M")
		#Minv = sp.coo_matrix(Minv)

		#startTimer("Multiply Minv")
		#Q = A * Minv
		
		#eval, evec = sp.linalg.eigs(A=Q, k=N_EVAL, which='SM')

		if USE_FEAST:
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

		else:
			startTimer("solve eigs")
			eval, evec = sp.linalg.eigs(A=A, k=N_EVAL, M=M, which='SM')
			endTimer()
			
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
			np.save(Eigenvectorsoutarray, evec)
			print("Eigenvector Analysis File Saved!")
						
