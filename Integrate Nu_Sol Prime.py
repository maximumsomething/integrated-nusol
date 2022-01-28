#-------1-------#
import numpy as np
import subprocess
import sys
import operator
import os
import os.path
import time
from scipy.linalg import solve
from datetime import datetime
import scipy.optimize as op
import scipy.sparse as sp
import numpy as np
#from mpl_toolkits.mplot3d import axes3d
#import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

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

USE_FEAST=True


global timerName
timerName = ""
timerStart = 0
def startTimer(name):
    global timerName
    global timerStart
    if timerName != "":
        endTimer()
    timerName = name
    timerStart = time.clock()

def endTimer():
    global timerName
    global timerStart
    timerEnd = time.clock()
    print(timerName, " took ", timerEnd - timerStart, "s")
    timerName = ""



#-------4-------#
def generate(ProjectName, NDIM, XMIN=0.0, XMAX=0.0, XDIV=0, XLEVEL = 0.0, YMIN=0.0, YMAX=0.0, YDIV=0, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=0, ZLEVEL = 0.0, Analytic = False, UserFunction = "", Overwrite = False):
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
        if Overwrite == True:
            try:
                myfile = open(PotentialArrayPath, "w")
                myfile.close()
                myfile = open(GenerateInfofile, "w")
                myfile.close()
            except IOError:
                print("The file(s) you are trying to save to are currently open. Close the file(s) and rerun the program again.")
                sys.exit()
            
        elif Overwrite == False:
            file_exists1 = os.path.exists(PotentialArrayPath)
            file_exists2 = os.path.exists(GenerateInfofile)
            if file_exists1 == True or file_exists2 == True:
                print("Overwrite failsafe has been set to false. A Potential Array File or Generating Into File has already been created with that name. Change the name of the file.")
                sys.exit()
                
    #-------4.3-------#
        if Analytic == False:
            if NDIM == 1:
                Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
                hz = Zgrid[1] - Zgrid[0]
                for zval in Zgrid:
                    LJ=0
                    for atom in atoms:
                        jointsigma = (atom.sigma + hydrogensigma)/2
                        jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                        magnitude = np.sqrt((zval-atom.z)**2+(XLEVEL-atom.x)**2+(YLEVEL-atom.y)**2)
                        LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                        LJ += LJpointval
                    LJPOL = np.append(LJ, LJPOL)
                LJPOL=np.reshape(LJPOL, (ZDIV))
                V =(LJPOL)
            if NDIM == 2:
                Xgrid = np.linspace(XMIN, XMAX, XDIV)
                hx = Xgrid[1] - Xgrid[0]
                Ygrid = np.linspace(YMIN, YMAX, YDIV)
                hy = Ygrid[1] - Ygrid[0]
                if hx!=hy:
                    print("WARNING! GRID SPACING IS UNEVEN! NUSOL WILL FAIL TO CONVERGE")
                for xval in Xgrid:
                     for yval in Ygrid:
                        LJ=0
                        for atom in atoms:
                            jointsigma = (atom.sigma + hydrogensigma)/2
                            jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                            magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(ZLEVEL-atom.z)**2)
                            LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                            LJ += LJpointval
                        LJPOL = np.append(LJ, LJPOL)
                LJPOL=np.reshape(LJPOL, (XDIV,YDIV))
                V =(LJPOL)
            if NDIM == 3:
                nZMIN = XMIN
                nZMAX = XMAX
                Xgrid = np.linspace(XMIN, XMAX, XDIV)
                hx = Xgrid[1] - Xgrid[0]
                Ygrid = np.linspace(YMIN, YMAX, YDIV)
                hy = Ygrid[1] - Ygrid[0]
                Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
                nZgrid = np.linspace(nZMIN, nZMAX, ZDIV)
                hz = nZgrid[1] - nZgrid[0]
                if (hx != hy) or (hx != hz) or (hy != hz):
                    print("WARNING! GRID SPACING IS UNEVEN! NUSOL WILL FAIL TO CONVERGE")
                for xval in Xgrid:
                     for yval in Ygrid:
                          for zval in Zgrid:
                            LJ=0
                            for atom in atoms:
                                jointsigma = (atom.sigma + hydrogensigma)/2
                                jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                                magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(zval-atom.z)**2)
                                LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                                LJ += LJpointval
                            LJPOL = np.append(LJ, LJPOL)
                LJPOL=np.reshape(LJPOL, (XDIV,YDIV,ZDIV))
                V =(LJPOL)
                
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
        print(V)
        
        if np.isnan(np.sum(V)) == False and np.isinf(np.sum(V)) == False:
            print("Maximum potential:", np.amax(V), "\nMinimum potential:", np.amin(V), "\nMinimum potential's array position", np.unravel_index(np.argmin(V, axis=None), V.shape))
        
            result = (np.where(V == np.amin(V)))
            min_list = []
            
    #-------4.6-------#
            
            if NDIM == 1:
                listofcoordinates = list(zip(ZMAX-result[0]*hz))
                for coord in listofcoordinates:
                    min_list.append(coord)

                print("The z position of the minimum is", (min_list))
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
                listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy))
            
                for coord in listofcoordinates:
                    min_list.append(coord)

                print("The x,y position of the minimum is", (min_list))


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
                listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy, ZMAX-result[2]*hz))
                for coord in listofcoordinates:
                    min_list.append(coord)
                print("The x,y,z position of the minimum is", (min_list))
                minimumpot = np.amin(V)
                xresult = result[0]
                yresult = result[1]
                zresult = result[2]
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
                print >>f, "ProjectName = %s NDIM = %d XMIN = %.8f XMAX = %.8f XDIV = %d XLEVEL = %.8f YMIN = %.8f YMAX = %.8f YDIV = %d YLEVEL = %.8f ZMIN = %.8f ZMAX = %.8f ZDIV = %d ZLEVEL = %.8f Analytic = %s UserFunction = %s Overwrite = %s MAXPOT = %.8f MINPOT = %.8f XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f ZSECONDDERIVATIVE = %.8f" % (ProjectName,NDIM,XMIN,XMAX,XDIV,XLEVEL,YMIN,YMAX,YDIV,YLEVEL,ZMIN,ZMAX,ZDIV,ZLEVEL,Analytic,UserFunction,Overwrite, np.amax(V), np.amin(V), xsecondderivative, ysecondderivative, zsecondderivative)
            elif np.isnan(np.sum(V)) == True or np.isinf(np.sum(V)) == True:
                print >>f, "ProjectName = %s NDIM = %d XMIN = %.8f XMAX = %.8f XDIV = %d XLEVEL = %.8f YMIN = %.8f YMAX = %.8f YDIV = %d YLEVEL = %.8f ZMIN = %.8f ZMAX = %.8f ZDIV = %d ZLEVEL = %.8f Analytic = %s UserFunction = %s Overwrite = %s MAXPOT = DNE MINPOT = DNE XSECONDDERIVATIVE = DNE YSECONDDERIVATIVE = DNE ZSECONDDERIVATIVE = DNE" % (ProjectName,NDIM,XMIN,XMAX,XDIV,XLEVEL,YMIN,YMAX,YDIV,YLEVEL,ZMIN,ZMAX,ZDIV,ZLEVEL,Analytic,UserFunction,Overwrite)
            f.close()
        except IOError:
            print("Error: The potential did not save. The file you wanted to save to was already opened. Close the file and rerun the program.")
            sys.exit()

        return V

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

            for iN in xrange(Nx):
                for iK in xrange(Ny):
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
            for iL in xrange(Nz):
              # process l-1 block
              # NumerovMatrix3D.append([ FORTRANoffset + iLx + iNx + iKx , FORTRANoffset + iLy + iNy + iKy - 1 ,  1.0 * self.cfg.preFactor3D , 0.0 ) )
                if (iL - 1 >= 0):
                    iLx = (iL) * Ny * Nx
                    iLy = (iL - 1) * Ny * Nx
                    for iN in xrange(Nx):
                        for iK in xrange(Ny):
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
                for iN in xrange(Nx):
                    for iK in xrange(Ny):
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
                    for iN in xrange(Nx):
                        for iK in xrange(Ny):
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
            print   >>f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % \
                  (matsize, A.getnnz(), XDIV, YDIV, ZDIV, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX, hz, hz, hz)
            # https://stackoverflow.com/a/4319159
            for row, col, data in zip(A.row, A.col, A.data):
                print   >>f,"%12d%12d % 18.16E % 18.16E" % (row + 1, col + 1, data, M[row, col])

            f.close()

            startTimer("call FEAST")

            p = subprocess.Popen('%s %f %f %d %s %s %s' % (FEAST_COMMAND,FEAST_E_MIN,FEAST_E_MAX,FEAST_M,FEAST_MATRIX_OUT_PATH,EIGENVALUES_OUT,EIGENVECTORS_OUT),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print (line,)
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
                print >> f, "%.12f" % (e)
            f.close()

            f = open(EIGENVECTORS_OUT,'w')
            for e in evec:
              line=''
              for i in e:
                line+="%.12e " % i
              print >> f, line
            f.close()
            print("Save complete!")
            print("Saving Eigenvector Analysis File...")
            np.save(Eigenvectorsoutarray, evec)
            print("Eigenvector Analysis File Saved!")
        
            
#-------6-------#                 
#generate("splittest", 1, 0.0, 0.0, 0, 13.0, 0.0, 0.0, 0, 13.0, -4.0, 4.0, 15, Overwrite = True)
numerov("matrixtesting3D", 3, -1.0, 1.0, 15, 0.0, -1.0, 1.0, 15, 0.0, 3.32, 5.32, 15, 0.0, N_EVAL = 3, Overwrite=False)
