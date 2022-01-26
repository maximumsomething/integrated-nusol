#!/usr/bin/python2.7
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
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)




class atom:
    def __init__ (self, x, y, z, charge=0, sigma=0, epsilon=0, mass=0):
        self.x=x
        self.y=y
        self.z=z
        self.charge=charge
        self.sigma=sigma
        self.epsilon=epsilon
        self.mass=mass

hydrogensigma = 2.571
hydrogenepsilon = 0.0000701127
Ck = 8.9875517923E9
alpha = 1


LJPOL = np.array([])


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




def Contour(ProjectName, Z, MINLEV=-.15, MAXLEV=.08, DIV=25):
    if type(Z) != float:
        print("Z is not in float format.")
    elif type(MINLEV) != float:
        print("MINLEV is not in float format.")
    elif type(MAXLEV) != float:
        print("MAXLEV is not in float format.")
    elif type(DIV) != int:
        print("DIV is not in an integer format. Make sure there are no decimals.")
    elif DIV <=0:
        print("DIV cannot be less than or equal to zero.")
    else: 
        try:
            myfile = open("generateinfo%s3D.dat" %ProjectName, "r")
        except:
            print("The designated file does not exist or is not in the right place.")
            sys.exit()
        lines = myfile.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
        XMIN = (float(line[11]))
        XMAX = (float(line[14]))
        XDIV = (int(line[2]))
        YMIN = (float(line[17]))
        YMAX = (float(line[20]))
        YDIV = (int(line[5]))
        xgrid = np.linspace(XMIN, XMAX, XDIV)
        ygrid = np.linspace(YMIN, YMAX, YDIV)
        XSTEPSIZE = (XMAX-XMIN)/(XDIV-1)
        YSTEPSIZE = (YMAX-YMIN)/(YDIV-1)
        meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")
        LJ = 0
        for atom in atoms:
            jointsigma = (atom.sigma + hydrogensigma)/2
            jointepsilon = np.sqrt((atom.epsilon * hydrogenepsilon))
            magnitude = np.sqrt((meshx-atom.x)**2+(meshy-atom.y)**2+(Z-atom.z)**2)
            LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
            LJ += LJpointval
        LJ = np.reshape(LJ, (XDIV, YDIV))
        print("Minmum potential is ", np.amin(LJ))
        print("The minimum potential's array position is:", np.unravel_index(np.argmin(LJ, axis=None), LJ.shape))

        result = np.where(LJ == np.amin(LJ))

        listofcoords = list(zip(YMIN+result[0]*YSTEPSIZE, XMIN+result[1]*XSTEPSIZE))

        min_list = []
        for coord in listofcoords:
            min_list.append(coord)

        print("The y,x position of the minimum is", (min_list))
        breaks = np.linspace(MINLEV, MAXLEV, DIV)
        CS1 = plt.contour(meshx, meshy, LJ, [breaks])
        plt.clabel(CS1, fontsize = 6)
        plt.show()

def Heat(ProjectName, Z):
    if type(Z) != float:
        print("Z is not in float format.")
    else:
        try:
            myfile = open("generateinfo%s3D.dat" %ProjectName, "r")
        except:
            print("The designated file does not exist or is not in the right place.")
            sys.exit()
        lines = myfile.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
        XMIN = (float(line[11]))
        XMAX = (float(line[14]))
        XDIV = (int(line[2]))
        YMIN = (float(line[17]))
        YMAX = (float(line[20]))
        YDIV = (int(line[5]))
        xgrid = np.linspace(XMIN, XMAX, XDIV)
        ygrid = np.linspace(YMIN, YMAX, YDIV)
        XSTEPSIZE = (XMAX-XMIN)/(XDIV-1)
        YSTEPSIZE = (YMAX-YMIN)/(YDIV-1)
        meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")
        LJ = 0
        for atom in atoms:
            jointsigma = (atom.sigma + hydrogensigma)/2
            jointepsilon = np.sqrt((atom.epsilon * hydrogenepsilon))
            magnitude = np.sqrt((meshx-atom.x)**2+(meshy-atom.y)**2+(Z-atom.z)**2)
            LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
            LJ += LJpointval
            print(LJ)
        LJ = np.reshape(LJ, (XDIV, YDIV))
        print("Minmum potential is ", np.amin(LJ))
        print("The minimum potential's array position is:", np.unravel_index(np.argmin(LJ, axis=None), LJ.shape))

        result = np.where(LJ == np.amin(LJ))

        listofcoords = list(zip(YMIN+result[0]*YSTEPSIZE, XMIN+result[1]*XSTEPSIZE))
        
        min_list = []
        for coord in listofcoords:
            min_list.append(coord)
        print("The y,x position of the minimum is", (min_list))
        
        fig,ax = plt.subplots()
        ax.pcolormesh(meshx, meshy, LJ)
        ax.set_aspect('equal')
        plt.show()


def Surface(ProjectName, Z):
    if type(Z) != float:
        print("Z is not in float format.")
    else:
        try:
            myfile = open("generateinfo%s3D.dat" %ProjectName, "r")
        except:
            print("The designated file does not exist or is not in the right place.")
            sys.exit()
        lines = myfile.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
        XMIN = (float(line[11]))
        XMAX = (float(line[14]))
        XDIV = (int(line[2]))
        YMIN = (float(line[17]))
        YMAX = (float(line[20]))
        YDIV = (int(line[5]))
        xgrid = np.linspace(XMIN, XMAX, XDIV)
        ygrid = np.linspace(YMIN, YMAX, YDIV)
        XSTEPSIZE = (XMAX-XMIN)/(XDIV-1)
        YSTEPSIZE = (YMAX-YMIN)/(YDIV-1)
        meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")
        LJ = 0
        for atom in atoms:
            jointsigma = (atom.sigma + hydrogensigma)/2
            jointepsilon = np.sqrt((atom.epsilon * hydrogenepsilon))
            magnitude = np.sqrt((meshx-atom.x)**2+(meshy-atom.y)**2+(Z-atom.z)**2)
            LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
            LJ += LJpointval
        LJ = np.reshape(LJ, (XDIV, YDIV))
        print("Minmum potential is ", np.amin(LJ))
        print("The minimum potential's array position is:", np.unravel_index(np.argmin(LJ, axis=None), LJ.shape))

        result = np.where(LJ == np.amin(LJ))

        listofcoords = list(zip(YMIN+result[0]*YSTEPSIZE, XMIN+result[1]*XSTEPSIZE))
        
        min_list = []
        for coord in listofcoords:
            min_list.append(coord)
        print("The y,x position of the minimum is", (min_list))
        figsur = plt.figure()
        axsur = figsur.add_subplot(111, projection='3d')
        axsur.plot_surface(meshx, meshy, LJ)
        plt.show()
        
                


##CHECK###
###failsafe needed###
def PotentialZGraphics(ProjectName, X, Y):
    if type(Y) != float:
        print("Y is not in float format.")
    elif type(X) != float:
        print("X is not in float format.")
    else:
        try:
            myfile = open("generateinfo%s3D.dat" %ProjectName, "r")
        except:
            print("The designtated file does not exist or is not in the right place.")
        lines = myfile.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
        ZDIV = (float(line[8]))
        ZMIN = (float(line[23]))
        ZMAX = (float(line[26]))
        myfile.close()
        zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
        print(zgrid)
        LJPOL = np.array([])
    ####CHECK IF THIS IS RIGHT!!!!####
        zgridprime = np.linspace(ZMAX, ZMIN, ZDIV) 
        for zval in zgrid:
                LJ=0
                for atom in atoms:
                    jointsigma = (atom.sigma + hydrogensigma)/2
                    print
                    jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                    magnitude = np.sqrt((X-atom.x)**2+(Y-atom.y)**2+(zval-atom.z)**2)
                    LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                    LJ += LJpointval

                LJPOL = np.append(LJ, LJPOL)

                
        Utot = LJPOL
                
        print("Maximum potential is", np.amax(Utot))
        print("Minimum potential is", np.amin(Utot))
        print(Utot)
        plt.plot(zgridprime, Utot)
        plt.show()


def generate1D(ProjectName, ZMIN, ZMAX, ZDIV, XLEVEL, YLEVEL, Analytic = False, UserFunction = "", Overwrite = False):
    ###FAILSAFES###
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(ZMIN) != float:
        print("ZMIN is not in a float format.")
    elif type(ZMAX) != float:
        print("ZMAX is not in a float format.")
    elif type(ZDIV) != int:
        print("ZDIV is not in an integer format. Make sure there is no decimal places for this input.")
    elif ZDIV <= 0:
        print("ZDIV cannot be less than or equal to zero.")
    elif type(XLEVEL) != float:
        print("XLEVEL is not in a float format.")
    elif type(YLEVEL) != float:
        print("YLEVEL is not in a float format.")
    elif ZMIN>=ZMAX:
        print("ZMIN cannot be greater or equal to ZMAX.")
    elif type(Analytic) != bool:
        print("Analytic is not in a boolean format. Make sure it is either true or false.")
    elif type(UserFunction) != str and Analytic == True:
            print("Function is not in a string format. Make sure the function is in quotation marks and contains only approproiate characters.")
    elif type(Overwrite) != bool:
        print("Overwrite is not in a boolean format. Make sure it is either true or false.")
        sys.exit()
    ###############
    else:
        print("Generating Potential...")
        if Overwrite == True:
            PotentialArrayPath = "Potential%s1D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s1D.dat" %(ProjectName)

            try:
                myfile = open(PotentialArrayPath, "w")
                myfile.close()
            except IOError:
                print("The potential array path you were trying to save to is currently open. Close the file and rerun the program again.")
                sys.exit()
            try:
                myfile = open(GenerateInfofile, "w")
                myfile.close()
            except IOError:
                print("The generating info file you were trying to save to is currently open. Close the file and rerun the program again.")
                sys.exit()
            
        elif Overwrite == False:
            PotentialArrayPath = "Potential%s1D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s1D.dat" %(ProjectName)
            file_exists1 = os.path.exists(PotentialArrayPath)
            file_exists2 = os.path.exists(GenerateInfofile)
            if file_exists1 == True or file_exists2 == True:
                print("Overwrite failsafe has been set to false. A Potential Array File or Generating Into File has already been created with that name. Change the name of the file.")
                sys.exit()
                
        if Analytic == False:
        

            LJPOL = np.array([])

            infcounter = 0
             
            Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
            hz = Zgrid[1] - Zgrid[0]
            for zval in Zgrid:
                LJ=0
                for atom in atoms:
                    jointsigma = (atom.sigma + hydrogensigma)/2
                    jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                    magnitude = np.sqrt((zval-atom.z)**2+(XLEVEL-atom.x)**2+(YLEVEL-atom.y)**2)
                    if magnitude == 0:
                        infcounter = infcounter + 1
                    LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                    LJ += LJpointval

                LJPOL = np.append(LJ, LJPOL)

            LJPOL=np.reshape(LJPOL, (ZDIV))

            Utot =(LJPOL)
        elif Analytic == True:
            try:
                Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
                z = Zgrid
                Utot = np.array(eval(UserFunction))
                hz = Zgrid[1] - Zgrid[0]
                infcounter = False
                if np.isnan(np.sum(Utot)) == True:
                    infcounter = True
                if np.isinf(np.sum(Utot)) == True:
                    infcounter = True
                 
            except NameError:
                print("Invalid function. Make sure your function is a function of z and that all non-elementary operations are preceded by 'np.'")
                sys.exit()
        print("###########################")
        print("Done generating potential!")
        print("###########################")
        print(Utot)
        
        if infcounter > 0 or infcounter == True:
            print("Maximum potential is undefined.")
            print("Minimum potential is undefinted.")
            print("The minimum potential's array position is undefined due to non-numeric values.")
            print("The z position of the minimum does not exist due to non-numeric values.")
        else:
            print("Maximum potential is", np.amax(Utot))
          
            print("Minimum potential is", np.amin(Utot))

            print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))

        
            result = (np.where(Utot == np.amin(Utot)))
        

            listofcoordinates = list(zip(ZMAX-result[0]*hz))
            min_list = []
        
            for coord in listofcoordinates:
                min_list.append(coord)

            print("The z position of the minimum is", (min_list))

            minimumpot = np.amin(Utot)

            print(Utot)
        
        
            zresult = result[0]
            try:
                zsecondderivative = ((Utot[zresult+1] - 2*minimumpot + Utot[zresult-1])/(hz**2))
                print("The second partial derivative with respect to z is", zsecondderivative)
            except:
                print("Undefined second partial derivative with respect to z.")
                zsecondderivative = float("Nan")


        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)

        np.save(PotentialArrayPath, Utot)
        print("###########################")
        
        try:
            f = open(GenerateInfofile, 'w')
            if infcounter == 0 or infcounter == False:
                print >>f, "ZDIV = %d ZMIN = %.8f ZMAX = %.8f XLEVEL = %.8f YLEVEL = %.8f, Analytic = %s , UserFunction = %s MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = %d, MINCOORDINATE =(%.8f, %.8f,%.8f), ZSECONDDERIVATIVE = %.8f" % (ZDIV,ZMIN,ZMAX,XLEVEL,YLEVEL, Analytic, UserFunction, np.amax(Utot), np.amin(Utot), result[0][0], XLEVEL, YLEVEL, min_list[0][0], zsecondderivative)
            elif infcounter == True or infcounter > 0:
                print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")
                print >>f, "ZDIV = %d ZMIN = %.8f ZMAX = %.8f XLEVEL = %.8f YLEVEL = %.8f, Analytic = %s , UserFunction = %s MAXPOT = Undefined, MINPOT = Undefined, MINARRAYPOSITON = Undefined , MINCOORDINATE = DOES NOT EXIST, ZSECONDDERIVATIVE = Undefined" % (ZDIV,ZMIN,ZMAX,XLEVEL,YLEVEL, Analytic, UserFunction)
            f.close()
        except IOError:
            print("Error: The potential did not save. The file you wanted to save to was already opened. Close the file and rerun the program.")
            sys.exit()
            
            
            

def generate2D(ProjectName, XMIN, XMAX, XDIV, YMIN, YMAX, YDIV, ZLEVEL, Analytic = False, UserFunction = "", Overwrite = False):
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(XMIN) != float:
        print("XMIN is not in a float format.") 
    elif type(XMAX) != float:
        print("XMAX is not in a float format.")
    elif type(YMIN) != float:
        print("YMIN is not in a float format.")
    elif type(YMAX) != float:
        print("YMAX is not in a float format.")
    elif type(YDIV) != int:
        print("YDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(XDIV) != int:
        print("XDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif XDIV <= 0:
        print("XDIV cannot be less than or equal to zero.")
    elif YDIV <= 0:
        print("YDIV cannot be less than or equal to zero.")
    elif type(ZLEVEL) != float:
        print("ZLEVEL is not in a float format.")
    elif XMIN>=XMAX:
        print("XMIN cannot be greater or equal to XMAX.")
    elif YMIN>=YMAX:
        print("YMIN cannot be greater or equal to XMAX.")
    elif type(Analytic) != bool:
        print("Analytic is not in a boolean format. Make sure it is either true or false.")
    elif type(UserFunction) != str and Analytic == True:
            print("Function is not in a string format. Make sure the function is in quotation marks and contains only approproiate characters.")
    elif type(Overwrite) != bool:
        print("Overwrite is not in a boolean format. Make sure it is either true or false.")
    ###############
 
    else:
        print("Generating Potential...")
        if Overwrite == True:
            PotentialArrayPath = "Potential%s2D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s2D.dat" %(ProjectName)

            try:
                myfile = open(PotentialArrayPath, "w")
                myfile.close()
            except IOError:
                print("The potential array path you were trying to save to is currently open. Close the file and rerun the program again.")
                sys.exit()
            try:
                myfile = open(GenerateInfofile, "w")
                myfile.close()
            except IOError:
                print("The generating info file you were trying to save to is currently open. Close th efile and rerun the program again.")
                sys.exit()
        elif Overwrite == False:
            PotentialArrayPath = "Potential%s2D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s2D.dat" %(ProjectName)
            file_exists1 = os.path.exists(PotentialArrayPath)
            file_exists2 = os.path.exists(GenerateInfofile)
            if file_exists1 == True or file_exists2 == True:
                print("Overwrite failsafe has been set to false. A Potential Array File or Generating Into File has already been created with that name. Change the name of the file.")
                sys.exit()
        if Analytic == False:
            
            LJPOL = np.array([])
            infcounter = 0
            
            Xgrid = np.linspace(XMIN, XMAX, XDIV)
            hx = Xgrid[1] - Xgrid[0]
            Ygrid = np.linspace(YMIN, YMAX, YDIV)
            hy = Ygrid[1] - Ygrid[0]
            for xval in Xgrid:
                 for yval in Ygrid:
                    LJ=0
                    for atom in atoms:
                        jointsigma = (atom.sigma + hydrogensigma)/2
                        jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                        magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(ZLEVEL-atom.z)**2)
                        if magnitude == 0:
                            infcounter = infcounter + 1
                        LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                        LJ += LJpointval

                    LJPOL = np.append(LJ, LJPOL)

            LJPOL=np.reshape(LJPOL, (XDIV,YDIV))

            Utot =(LJPOL)
####WARNING CHECK IF RIGHT XYZ AXIS!!!!####
        elif Analytic == True:
            try:
                Xgrid = np.linspace(XMIN, XMAX, XDIV)
                hx = Xgrid[1] - Xgrid[0]
                Ygrid = np.linspace(YMIN, YMAX, YDIV)
                hy = Ygrid[1] - Ygrid[0]
                x,y = np.meshgrid(Xgrid,Ygrid)
                print(UserFunction)
                Utot = np.array(eval(UserFunction))
                infcounter = False
                if np.isnan(np.sum(Utot)) == True:
                    infcounter = True
                if np.isinf(np.sum(Utot)) == True:
                    infcounter = True
            except NameError:
                print("Invalid function. Make sure your function is a function of x and y and that all non-elementary operations are proceded by 'np.'")
                sys.exit()
                
        print("###########################")
        print("Done generating potential!")
        print("###########################")
        print(Utot)

        if infcounter > 0 or infcounter == True:
            print("Maximum potential is undefined.")
            print("Minimum potential is undefined.")
            print("The minimum potential's array position is undefined due to non-numeric values.")
            print("The x, y position of the minimum does not exist due to non-numeric values.")
        else:
            
            print("Maximum potential is", np.amax(Utot))
              
            print("Minimum potential is", np.amin(Utot))

            print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))


            result = (np.where(Utot == np.amin(Utot)))

            listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy))

            min_list = []
            
            for coord in listofcoordinates:
                min_list.append(coord)

            print("The x,y position of the minimum is", (min_list))


            minimumpot = np.amin(Utot)

            print(Utot)
            

            xresult = result[0]
            yresult = result[1]

            try:
                xsecondderivative = ((Utot[xresult+1, yresult] - 2*minimumpot + Utot[xresult-1, yresult])/(hx**2))
                print("The second partial derivative with respect to x is", xsecondderivative)
            except:
                print("Undefined second partial derivative with respect to x.")
                xsecondderivative = float("Nan")
            try:
                ysecondderivative = ((Utot[xresult, yresult+1] - 2*minimumpot + Utot[xresult, yresult-1])/(hy**2))
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
        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)
        np.save(PotentialArrayPath, Utot)
        print("###########################")
        try:
            f = open(GenerateInfofile, 'w')
            if infcounter == 0 or infcounter == False:
                print >>f, "XDIV = %d YDIV = %d XMIN = %.8f XMAX = %.8f YMIN = %.8f YMAX = %.8f ZLEVEL = %.8f MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = (%d, %d), MINCOORDINATE = (%.8f, %.8f,%.8f), XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f DELSQUARED = %.8f, Analytic = %s, UserFunction = %s " % (XDIV,YDIV,XMIN,XMAX,YMIN,YMAX,ZLEVEL, np.amax(Utot), np.amin(Utot), xresult, yresult, min_list[0][0], min_list[0][1], ZLEVEL, xsecondderivative, ysecondderivative, delsquared, Analytic, UserFunction)
            elif infcounter == True or infcounter > 0:
                print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")
                print >>f, "XDIV = %d YDIV = %d XMIN = %.8f XMAX = %.8f YMIN = %.8f YMAX = %.8f ZLEVEL = %.8f MAXPOT = Undefined, MINPOT = Undefined, MINARRAYPOSITON = Undefined, MINCOORDINATE = DOES NOT EXIST, XSECONDDERIVATIVE = Undefined YSECONDDERIVATIVE = Undefined DELSQUARED = Undefined, Analytic = %s, UserFunction = %s " % (XDIV,YDIV,XMIN,XMAX,YMIN,YMAX,ZLEVEL,Analytic, UserFunction)
            f.close()
        except IOError:
            print("Error: The potential did not save. The file you wanted to save to was already opened. Close the file and rerun the program.")
            sys.exit()
    
def generate3D(ProjectName, XMIN, XMAX, XDIV, YMIN, YMAX, YDIV, ZMIN, ZMAX, ZDIV, Analytic = False, UserFunction = "", Overwrite = False): 
    ###INPUT FAILSAFES###
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(XMIN) != float:
        print("XMIN is not in a float format.") 
    elif type(XMAX) != float:
        print("XMAX is not in a float format.")
    elif type(YMIN) != float:
        print("YMIN is not in a float format.")
    elif type(YMAX) != float:
        print("YMAX is not in a float format.")
    elif type(ZMIN) != float:
        print("ZMIN is not in a float format.") 
    elif type(ZMAX) != float:
        print("ZMAX is not in a float format.")
    elif type(YDIV) != int:
        print("YDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(XDIV) != int:
        print("XDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(ZDIV) != int:
        print("ZDIV is not in an integer format. Maker sure there are no decimal places for this input.")
    elif XDIV <= 0:
        print("XDIV cannot be less than or equal to zero.")
    elif YDIV <= 0:
        print("YDIV cannot be less than or equal to zero.")
    elif ZDIV <= 0:
        print("ZDIV cannot be less than or equal to zero.")
    elif XMIN>=XMAX:
        print("XMIN cannot be greater or equal to XMAX.")
    elif YMIN>=YMAX:
        print("YMIN cannot be greater or equal to YMAX.")
    elif ZMIN>=ZMAX:
        print("ZMIN cannot be greater or equal to ZMAX.")
    elif type(Analytic) != bool:
        print("Analytic is not in a boolean format. Make sure it is either true or false.")
    elif type(UserFunction) != str and Analytic == True:
            print("Function is not in a string format. Make sure the function is in quotation marks and contains only approproiate characters.")
    elif type(Overwrite) != bool:
        print("Overwrite is not in a boolean format. Make sure it is either true or false.")
    else:


        print("Generating Potential...")
        if Overwrite == True:
            PotentialArrayPath = "Potential%s3D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s3D.dat" %(ProjectName)
            try:
                myfile = open(PotentialArrayPath, "w")
                myfile.close()
            except IOError:
                print("The potential array path you were trying to save to is currently open. Close the file and rerun the program again.")
                sys.exit()
            try:
                myfile = open(GenerateInfofile, "w")
                myfile.close()
            except IOError:
                print("The generating info file you were trying to save to is currently open. Close th efile and rerun the program again.")
                sys.exit()
        elif Overwrite == False:
            PotentialArrayPath = "Potential%s3D.npy" %(ProjectName)
            GenerateInfofile = "generateinfo%s3D.dat" %(ProjectName)
            file_exists1 = os.path.exists(PotentialArrayPath)
            file_exists2 = os.path.exists(GenerateInfofile)
            if file_exists1 == True or file_exists2 == True:
                print("Overwrite failsafe has been set to false. A Potential Array File or Generating Into File has already been created with that name. Change the name of the file.")
                sys.exit()
        if Analytic == False:
            LJPOL = np.array([])
            nZMIN = XMIN
            nZMAX = XMAX

            infcounter = 0
             
            Xgrid = np.linspace(XMIN, XMAX, XDIV)
            hx = Xgrid[1] - Xgrid[0]
            Ygrid = np.linspace(YMIN, YMAX, YDIV)
            hy = Ygrid[1] - Ygrid[0]
            Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
            nZgrid = np.linspace(nZMIN, nZMAX, ZDIV)
            hz = nZgrid[1] - nZgrid[0]
            for xval in Xgrid:
                 for yval in Ygrid:
                      for zval in Zgrid:
                        LJ=0
                        for atom in atoms:
                            jointsigma = (atom.sigma + hydrogensigma)/2
                            jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                            magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(zval-atom.z)**2)
                            if magnitude == 0:
                                infcounter = infcounter + 1
                            LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                            LJ += LJpointval

                        LJPOL = np.append(LJ, LJPOL)

            LJPOL=np.reshape(LJPOL, (XDIV,YDIV,ZDIV))

            Utot =(LJPOL)
        elif Analytic == True:
            try:
                Xgrid = np.linspace(XMIN, XMAX, XDIV)
                hx = Xgrid[1] - Xgrid[0]
                Ygrid = np.linspace(YMIN, YMAX, YDIV)
                hy = Ygrid[1] - Ygrid[0]
                Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
                hz = Zgrid[1]-Zgrid[0]
                x,y,z = np.meshgrid(Xgrid, Ygrid, Zgrid)
                Utot = np.array(eval(UserFunction))
                infcounter = False
                if np.isnan(np.sum(Utot)) == True:
                    infcounter = True
                if np.isinf(np.sum(Utot)) == True:
                    infcounter = True
            except NameError:
                print("Invalid function. Make sure your function is a function of x and y and that all non-elementary operations are proceded by 'np.'")
                sys.exit()
        print("###########################")
        print("Done generating potential!")
        print("###########################")

        print("Maximum potential is", np.amax(Utot))
          
        print("Minimum potential is", np.amin(Utot))

        print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))


        result = (np.where(Utot == np.amin(Utot)))

        listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy, ZMAX-result[2]*hz))

        min_list = []
        
        for coord in listofcoordinates:
            min_list.append(coord)

        print("The x,y,z position of the minimum is", (min_list))


        minimumpot = np.amin(Utot)

        print(Utot)
        

        xresult = result[0]
        yresult = result[1]
        zresult = result[2]

        try:
            xsecondderivative = ((Utot[xresult+1, yresult, zresult] - 2*minimumpot + Utot[xresult-1, yresult, zresult])/(hx**2))
            print("The second partial derivative with respect to x is", xsecondderivative)
        except:
            print("Undefined second partial derivative with respect to x.")
            xsecondderivative = float("nan")
        try:
            ysecondderivative = ((Utot[xresult, yresult+1, zresult] - 2*minimumpot + Utot[xresult, yresult-1, zresult])/(hy**2))
            print("The second partial derivative with respect to y is", ysecondderivative)
        except:
            print("Undefined second partial derivative with respect to y.")
            ysecondderivative = float("nan")
        try:
            zsecondderivative = ((Utot[xresult, yresult, zresult+1] - 2*minimumpot + Utot[xresult, yresult, zresult-1])/(hz**2))
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
        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)

        np.save(PotentialArrayPath, Utot)
        print("###########################")

        f = open(GenerateInfofile,'w')
        print >>f , "XDIV = %d YDIV = %d ZDIV = %d XMIN = %.8f XMAX = %.8f YMIN = %.8f YMAX = %.8f ZMIN = %.8f ZMAX = %.8f MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = (%d, %d, %d), MINCOORDINATE = (%.8f, %.8f,%.8f), XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f ZSECONDDERIVATIVE = %.8f DELSQUARED = %.8f "% (XDIV,YDIV,ZDIV,XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX, np.amax(Utot), np.amin(Utot), xresult, yresult, zresult, min_list[0][0], min_list[0][1], min_list[0][2], xsecondderivative, ysecondderivative, zsecondderivative, delsquared)
        f.close()
        if infcounter > 0:
            print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")


def numerov1D(ProjectName, N_EVAL, ZMIN, ZMAX, ZDIV, XLEVEL, YLEVEL, HBAR = 1.0, MASS = 3672.31 ):
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(ZMIN) != float:
        print("ZMIN is not in a float format.")
    elif type(ZMAX) != float:
        print("ZMAX is not in a float format.")
    elif type(ZDIV) != float:
        print("ZDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif ZDIV <= 0:
        print("ZDIV cannot be less than or equal to zero.")
    elif type(XLEVEL) != float:
        print("XLEVEL is not in a float format.")
    elif type(YLEVEL) != float:
        print("YLEVEL is not in a float format.")
    elif ZMIN>=ZMAX:
        print("ZMIN cannot be greater or equal to ZMAX.")
    elif type(N_EVAL) != int:
        print("N_EVAL is not in an integer format. Maker sure there are no decimal places for this input.")
    elif N_EVAL < 0:
        print("N_EVAL cannot be less than zero.")
    elif MASS <= 0:
        print("MASS cannot be less than or equal to zero.")
    elif type(MASS) != float:
        print("MASS is not in a float format.")
    elif type(HBAR) != float:
        print("HBAR is not in a float format.")
    else:
        startTime = datetime.now()
        print("Generating Potential...")
        #####INPUTS#####
        #1DEigenvectorAnalysis = False
        #1DZstartvalue = 2.0
        #2DEigenvectorAnalysis = False
        #2DZstartvalue = 2.0



        EIGENVALUES_OUT = "valout%s1D.dat" %(ProjectName)
        EIGENVECTORS_OUT = "vecout%s1D.dat" %(ProjectName)
        PotentialArrayPath = "Potential%s1D.npy" %(ProjectName)
        Eigenvectorsoutarray = "Vecanalysis%s1D.npy" %(ProjectName)
        GenerateInfofile = "generateinfo%s1D.dat" %(ProjectName)


    ##fix user interface##

        LJPOL = np.array([])
        infcounter = 0
        
         
        Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
        hz = Zgrid[1] - Zgrid[0]
        for zval in Zgrid:
            LJ=0
            for atom in atoms:
                jointsigma = (atom.sigma + hydrogensigma)/2
                jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                magnitude = np.sqrt((zval-atom.z)**2+(XLEVEL-atom.x)**2+(YLEVEL-atom.y)**2)
                if magnitude == 0:
                    infcounter = infcounter + 1
                LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                LJ += LJpointval

            LJPOL = np.append(LJ, LJPOL)

        LJPOL=np.reshape(LJPOL, (ZDIV))

        Utot =(LJPOL)
        print("###########################")
        print("Done generating potential!")
        print("###########################")

        print("Maximum potential is", np.amax(Utot))
          
        print("Minimum potential is", np.amin(Utot))

        print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))


        result = (np.where(Utot == np.amin(Utot)))

        listofcoordinates = list(zip(ZMAX-result[0]*hz))

        min_list = []
        
        for coord in listofcoordinates:
            min_list.append(coord)

        print("The z position of the minimum is", (min_list))


        minimumpot = np.amin(Utot)

        print(Utot)

        zresult = result[0]

        try:
            zsecondderivative = ((Utot[zresult+1] - 2*minimumpot + Utot[zresult-1])/(hz**2))
            print("The second partial derivative with respect to z is", zsecondderivative)
        except:
            print("Undefined second partial derivative with respect to z.")
            zsecondderivative = float("nan")
        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)

        np.save(PotentialArrayPath, Utot)
        print("###########################")
        print("Loading the potential array...")
        print("###########################")
        V = np.load(PotentialArrayPath)
        print("Loaded the potential array!")

        f = open(GenerateInfofile, 'w')
        print >>f, "ZDIV = %d ZMIN = %.8f ZMAX = %.8f XLEVEL = %.8f YLEVEL = %.8f, N_EVAL = %d HBAR = %.8f MASS = %.8f MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = %d, MINCOORDINATE = (%.8f, %.8f, %.8f), ZSECONDDERIVATIVE = %.8f" % (ZDIV,ZMIN,ZMAX,XLEVEL,YLEVEL, N_EVAL, HBAR, MASS, np.amax(Utot), np.amin(Utot), result[0], XLEVEL, YLEVEL, min_list[0][0], zsecondderivative)
        f.close()
        if infcounter > 0:
            print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")
        preFactor1D = - 6.0 * HBAR * HBAR / (MASS * hz * hz)

        print ('Creating 1D Numerov Matrix -- %d grid points [Z] -- grid spacing %f' % (ZDIV, hz))
        NumerovMatrix1D = []
        FORTRANoffset = 1
        Nele = 0
        for i in xrange(ZDIV):
        ##cycles through divisions##
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
            ##puts A data into a matrix specified by its coordinates
        A = sp.coo_matrix((dataA, (row, col)), shape=(ZDIV, ZDIV))
        M = sp.csr_matrix((dataM, (row, col)), shape=(ZDIV, ZDIV))
                                                                                                         
    ####running ARPACK instead and calling WRITE_EVAL_AND_EVEC Function#####
        print("Starting up NuSol...")
        print ('Note: Using buildin SCIPY ARPACK interface for Numerov.')
        eval,evec = sp.linalg.eigs(A=A,k=N_EVAL,M=M,which='SM')
        print("NuSol completed successfully!")
        print("Saving Eigenvalue and Eigenvector Data Files...")
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
        current_dir_path = "C:\Users\Student\Desktop\NuSol_official"
        os.chdir(current_dir_path)
        print("Saving Eigenvector Analysis File...")
        np.save(Eigenvectorsoutarray, evec)
        print("Eigenvector Analysis File Saved!")


def numerov2D(ProjectName, N_EVAL, ZLEVEL, XMIN, XMAX, XDIV, YMIN, YMAX, YDIV, HBAR = 1.0, MASS = 3672.31 ):
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(XMIN) != float:
        print("XMIN is not in a float format.") 
    elif type(XMAX) != float:
        print("XMAX is not in a float format.")
    elif type(YMIN) != float:
        print("YMIN is not in a float format.")
    elif type(YMAX) != float:
        print("YMAX is not in a float format.")
    elif type(YDIV) != int:
        print("YDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(XDIV) != int:
        print("XDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif XDIV <= 0:
        print("XDIV cannot be less than or equal to zero.")
    elif YDIV <= 0:
        print("YDIV cannot be less than or equal to zero.")
    elif type(ZLEVEL) != float:
        print("ZLEVEL is not in a float format.")
    elif XMIN>=XMAX:
        print("XMIN cannot be greater or equal to XMAX.")
    elif YMIN>=YMAX:
        print("YMIN cannot be greater or equal to XMAX.")
    elif type(N_EVAL) != int:
        print("N_EVAL is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(HBAR) != int:
        print("HBAR is not in a float format.")
    elif type(MASS) != float:
        print("MASS is not in a float format.")
    elif N_EVAL < 0:
        print("N_EVAL cannot be less than zero.")
    elif MASS <= 0:
        print("MASS cannot be less than or equal to zero.")
    else:
        startTime = datetime.now()
        print("Generating Potential...")
        #####INPUTS#####
        #1DEigenvectorAnalysis = False
        #1DZstartvalue = 2.0
        #2DEigenvectorAnalysis = False
        #2DZstartvalue = 2.0



        EIGENVALUES_OUT = "valout%s2D.dat" %(ProjectName)
        EIGENVECTORS_OUT = "vecout%s2D.dat" %(ProjectName)
        PotentialArrayPath = "Potential%s2D.npy" %(ProjectName)
        Eigenvectorsoutarray = "Vecanalysis%s2D.npy" %(ProjectName)
        GenerateInfofile = "generateinfo%s2D.dat" %(ProjectName) 



        LJPOL = np.array([])
        infcounter = 0
         
        Xgrid = np.linspace(XMIN, XMAX, XDIV)
        hx = Xgrid[1] - Xgrid[0]
        Ygrid = np.linspace(YMIN, YMAX, YDIV)
        hy = Ygrid[1] - Ygrid[0]
        for xval in Xgrid:
             for yval in Ygrid:
                LJ=0
                for atom in atoms:
                    jointsigma = (atom.sigma + hydrogensigma)/2
                    jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                    magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(ZLEVEL-atom.z)**2)
                    if magnitude == 0:
                        infcounter = infcounter + 1
                    LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                    LJ += LJpointval

                LJPOL = np.append(LJ, LJPOL)

        LJPOL=np.reshape(LJPOL, (XDIV,YDIV))

        Utot =(LJPOL)
        print("###########################")
        print("Done generating potential!")
        print("###########################")

        print("Maximum potential is", np.amax(Utot))
          
        print("Minimum potential is", np.amin(Utot))

        print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))


        result = (np.where(Utot == np.amin(Utot)))

        listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy))

        min_list = []
        
        for coord in listofcoordinates:
            min_list.append(coord)

        print("The x,y position of the minimum is", (min_list))


        minimumpot = np.amin(Utot)

        print(Utot)

        xresult = result[0]
        yresult = result[1]

        try:
            xsecondderivative = ((Utot[xresult+1, yresult] - 2*minimumpot + Utot[xresult-1, yresult])/(hx**2))
            print("The second partial derivative with respect to x is", xsecondderivative)
        except:
            print("Undefined second partial derivative with respect to x.")
            xsecondderivative = float("nan")
        try:
            ysecondderivative = ((Utot[xresult, yresult+1] - 2*minimumpot + Utot[xresult, yresult-1])/(hy**2))
            print("The second partial derivative with respect to y is", ysecondderivative)
            ysecondderivative = float("nan")
        except:
            print("Undefined second partial derivative with respect to y.")
        try: 
            print("Del Squared is", xsecondderivative+ysecondderivative)
            delsquared = xsecondderivative+ysecondderivative
        except:
            print("Del Squared is undefined.")
            delsquared = float("nan")
        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)

        np.save(PotentialArrayPath, Utot)
        print("###########################")
        print("Loading the potential array...")
        print("###########################")
        V = np.load(PotentialArrayPath)
        print("Loaded the potential array!")
        f = open(GenerateInfofile,'w')
        print >>f , "XDIV = %d YDIV = %d XMIN = %.8f XMAX = %.8f YMIN = %.8f YMAX = %.8f ZLEVEL = %.8f N_EVAL = %d HBAR = .8f% MASS .8f% MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = (%d, %d), MINCOORDINATE = (%.8f, %.8f,%.8f), XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f DELSQUARED = %.8f " % (XDIV,YDIV,XMIN,XMAX,YMIN,YMAX,ZLEVEL, N_EVAL, HBAR, MASS, np.amax(Utot), np.amin(Utot), xresult, yresult, min_list[0][0], min_list[0][1], ZLEVEL, xsecondderivative, ysecondderivative, delsquared)
        f.close()
        if infcounter > 0:
            print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")
        preFactor2D =   1.0 / (MASS * hx * hx / (HBAR * HBAR))
        
           
        if (hx != hy):
            print "Check you gridspacing!"
        else: 
            print ('Creating 2D Numerov Matrix -- %dx%d=%d grid points [XY] -- grid spacing %f Bohr' % (XDIV,YDIV,XDIV*YDIV, hx))
            Nx = XDIV
            Ny = YDIV
            print Nx, Ny
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
                                                                                                               
    ####running ARPACK instead and calling WRITE_EVAL_AND_EVEC Function#####
        print("Starting up NuSol...")
        print ('Note: Using buildin SCIPY ARPACK interface for Numerov.')
        eval,evec = sp.linalg.eigs(A=A,k=N_EVAL,M=M,which='SM')
        print("NuSol completed successfully!")
        print("Saving Eigenvalue and Eigenvector Data Files...")
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
        current_dir_path = "C:\Users\Student\Desktop\NuSol_official"
        os.chdir(current_dir_path)
        print("Saving Eigenvector Analysis File...")
        np.save(Eigenvectorsoutarray, evec)
        print("Eigenvector Analysis File Saved!")

def numerov3D(ProjectName, N_EVAL, XMIN, XMAX, XDIV, YMIN, YMAX, YDIV, ZMIN, ZMAX, ZDIV, HBAR = 1.0, MASS = 3672.31):
    if type(ProjectName) != str:
        print("Project name is not in a string format. Make sure the Project name is in quotation marks and contains only appropriate characters.")
    elif type(XMIN) != float:
        print("XMIN is not in a float format.") 
    elif type(XMAX) != float:
        print("XMAX is not in a float format.")
    elif type(YMIN) != float:
        print("YMIN is not in a float format.")
    elif type(YMAX) != float:
        print("YMAX is not in a float format.")
    elif type(ZMIN) != float:
        print("ZMIN is not in a float format.") 
    elif type(ZMAX) != float:
        print("ZMAX is not in a float format.")
    elif type(YDIV) != int:
        print("YDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(XDIV) != int:
        print("XDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(ZDIV) != int:
        print("ZDIV is not in an integer format. Make sure there are no decimal places for this input.")
    elif XDIV <= 0:
        print("XDIV cannot be less than or equal to zero.")
    elif YDIV <= 0:
        print("YDIV cannot be less than or equal to zero.")
    elif ZDIV <= 0:
        print("ZDIV cannot be less than or equal to zero.")
    elif XMIN>=XMAX:
        print("XMIN cannot be greater or equal to XMAX.")
    elif YMIN>=YMAX:
        print("YMIN cannot be greater or equal to YMAX.")
    elif ZMIN>=ZMAX:
        print("ZMIN cannot be greater or equal to ZMAX.")
    elif type(HBAR) != float:
        print("HBAR is not in a float format.")
    elif type(N_EVAL) != int:
        print("N_EVAL is not in an integer format. Make sure there are no decimal places for this input.")
    elif type(MASS) != float:
        print("MASS is not in a float format.")
    elif N_EVAL < 0:
        print("N_EVAL cannot be less than zero.")
    elif MASS <= 0:
        print("MASS cannot be less than or equal to zero.")
    else:
        startTime = datetime.now()
        print("Generating Potential...")
        #####INPUTS#####
        #1DEigenvectorAnalysis = False
        #1DZstartvalue = 2.0
        #2DEigenvectorAnalysis = False
        #2DZstartvalue = 2.0



        EIGENVALUES_OUT = "valout%s3D.dat" %(ProjectName)
        EIGENVECTORS_OUT = "vecout%s3D.dat" %(ProjectName)
        PotentialArrayPath = "Potential%s3D.npy" %(ProjectName)
        Eigenvectorsoutarray = "Vecanalysis%s3D.npy" %(ProjectName)
        GenerateInfofile = "generateinfo%s3D.dat" %(ProjectName)


        LJPOL = np.array([])
        nZMIN = XMIN
        nZMAX = XMAX

        infcounter = 0
         
        Xgrid = np.linspace(XMIN, XMAX, XDIV)
        hx = Xgrid[1] - Xgrid[0]
        Ygrid = np.linspace(YMIN, YMAX, YDIV)
        hy = Ygrid[1] - Ygrid[0]
        Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
        nZgrid = np.linspace(nZMIN, nZMAX, ZDIV)
        hz = nZgrid[1] - nZgrid[0]
        for xval in Xgrid:
             for yval in Ygrid:
                  for zval in Zgrid:
                    LJ=0
                    for atom in atoms:
                        jointsigma = (atom.sigma + hydrogensigma)/2
                        jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
                        magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(zval-atom.z)**2)
                        if magnitude == 0:
                            infcounter = infcounter + 1
                        LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
                        LJ += LJpointval

                    LJPOL = np.append(LJ, LJPOL)

        LJPOL=np.reshape(LJPOL, (XDIV,YDIV,ZDIV))

        Utot =(LJPOL)
        print("###########################")
        print("Done generating potential!")
        print("###########################")

        print("Maximum potential is", np.amax(Utot))
          
        print("Minimum potential is", np.amin(Utot))

        print("The minimum potential's array position is:" , np.unravel_index(np.argmin(Utot, axis=None), Utot.shape))


        result = (np.where(Utot == np.amin(Utot)))

        listofcoordinates = list(zip(XMAX-result[0]*hx, YMAX-result[1]*hy, ZMAX-result[2]*hz))

        min_list = []
        
        for coord in listofcoordinates:
            min_list.append(coord)

        print("The x,y,z position of the minimum is", (min_list))


        minimumpot = np.amin(Utot)

        print(Utot)


        xresult = result[0]
        yresult = result[1]
        zresult = result[2]

        try:
            xsecondderivative = ((Utot[xresult+1, yresult, zresult] - 2*minimumpot + Utot[xresult-1, yresult, zresult])/(hx**2))
            print("The second partial derivative with respect to x is", xsecondderivative)
        except:
            print("Undefined second partial derivative with respect to x.")
            xsecondderivative = float("nan")
        try:
            ysecondderivative = ((Utot[xresult, yresult+1, zresult] - 2*minimumpot + Utot[xresult, yresult-1, zresult])/(hy**2))
            print("The second partial derivative with respect to y is", ysecondderivative)
        except:
            print("Undefined second partial derivative with respect to y.")
            ysecondderivative = float("nan")
        try:
            zsecondderivative = ((Utot[xresult, yresult, zresult+1] - 2*minimumpot + Utot[xresult, yresult, zresult-1])/(hz**2))
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
        print("###########################")
        print("Saving the potential array as", PotentialArrayPath)

        np.save(PotentialArrayPath, Utot)
        print("###########################")
        print("Loading the potential array...")
        print("###########################")
        V = np.load(PotentialArrayPath)
        print("Loaded the potential array!")
        f = open(GenerateInfofile,'w')
        print >>f , "XDIV = %d YDIV = %d ZDIV = %d XMIN = %.8f XMAX = %.8f YMIN = %.8f YMAX = %.8f ZMIN = %.8f ZMAX = %.8f N_EVAL = %d HBAR = %.8f MASS = %.8f MAXPOT = %.8f, MINPOT = %.8f, MINARRAYPOSITON = (%d, %d, %d), MINCOORDINATE = (%.8f, %.8f,%.8f), XSECONDDERIVATIVE = %.8f YSECONDDERIVATIVE = %.8f ZSECONDDERIVATIVE = %.8f DELSQUARED = %.8f "% (XDIV,YDIV,ZDIV,XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,N_EVAL, HBAR, MASS, np.amax(Utot), np.amin(Utot), xresult, yresult, zresult, min_list[0][0], min_list[0][1], min_list[0][2], xsecondderivative, ysecondderivative, zsecondderivative, delsquared)
        f.close()
        if infcounter > 0:
            print("WARNING! THIS GENERATED ARRAY CONTAINS NONNUMERICAL VALUES! RUNNING THIS ARRAY THROUGH NUSOL OR GRAPHICS INTERFACES MAY NOT WORK!")

        preFactor3D =  - (HBAR * HBAR) / (2.0 * MASS * hx * hx)
        
        
        if (hx != hy or hx != hz or hy!=hz):
            print "Check you gridspacing!"
        else:
            print ('Creating 3D Numerov Matrix -- %dx%dx%d=%d grid points [XYZ] -- grid spacing %f Bohr' % (XDIV,YDIV,ZDIV,XDIV*YDIV*ZDIV, hx))
            
            Nx = XDIV
            Ny = YDIV
            Nz = ZDIV
            NumerovMatrix3D = []
            FORTRANoffset = 1
            Nele = 0
            for iL in xrange(Nz):
      
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
            XDIV*YDIV*ZDIV, XDIV*YDIV*ZDIV))
            M = sp.csr_matrix((dataM, (row, col)), shape=(
            XDIV*YDIV*ZDIV, XDIV*YDIV*ZDIV))


                                                                                                            
    ####running ARPACK instead and calling WRITE_EVAL_AND_EVEC Function#####
        print("Starting up NuSol...")
        print ('Note: Using buildin SCIPY ARPACK interface for Numerov.')
        eval,evec = sp.linalg.eigs(A=A,k=N_EVAL,M=M,which='SM')
        print("NuSol completed successfully!")
        print("Saving Eigenvalue and Eigenvector Data Files...")
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
        current_dir_path = "C:\Users\Student\Desktop\NuSol_official"
        os.chdir(current_dir_path)
        print("Saving Eigenvector Analysis File...")
        np.save(Eigenvectorsoutarray, evec)
        print("Eigenvector Analysis File Saved!")


#####Eigenvector Analysis#
def ZEigenvectors(ProjectName, Eval):
    print("Starting 1D Eigenvector Analysis!")
    myfile = open("inputs%s3D.dat" %ProjectName, "r")
    lines = myfile.readlines()
    for line in lines:
        line = line.strip()
        line = line.split()
    NDIM = (int(line[10]))
    XMIN = (float(line[3]))
    XMAX = (float(line[4]))
    XDIV = (int(line[0]))
    YMIN = (float(line[5]))
    YMAX = (float(line[6]))
    YDIV = (int(line[2]))
    ZMIN = (float(line[7]))
    ZMAX = (float(line[8]))
    ZDIV = (int(line[6]))
    ZArray = np.array([])
    Evecs = np.load("Vecanalysis%s%s.npy" %(ProjectName, NDIM))
    
    

##        1DZArray = np.array([])
##        s = int((XMAX/hx))
##        B = np.array([])
##        1DEigenvectorsIn = np.load(Eigenvectorsoutarray)
##        for val in 1DEigenvectorsIn:
##            string_val = str(val)
##            parentremover = string_val.split('[')
##            keep = parentremover[1]
##            secondparentremover = keep.split(']')
##            nextkeep = secondparentremover[0]
##            realimg = nextkeep.split('+0.j')
##            realpart= float(realimg[0])
##            scientific_notation = "{:.4e}".format(realpart)
##            final = float(scientific_notation)
##    
##            B = np.append(B, final)
##        B = np.reshape(B, (XDIV, YDIV, ZDIV))
##        1DZArray = B[s:s+1,s:s+1]
##        1DZArray = np.resize(1DZArray, (ZDIV))
##        plt.plot(Zgrid, 1DZArray)
##        plt.show()
##        fig = plt.figure()
##        ax1 = fig.add_subplot(111, projection='3d')
##        print("Starting 2D Eigenvector Analysis!")
##        h = (2DZstart - ZMIN)/hz
##        2DZArray = np.array([])
##        meshx, meshy = np.meshgrid(Xgrid, Ygrid, sparse=False, indexing = "xy")
##        2DEigenvectorsIn = np.load(Eigenvectorsoutarray)
##        for val in 2DEigenvectorsIn:
##            string_val = str(val)
##            parentremover = string_val.split('[')
##            keep = parentremover[1]
##            secondparentremover = keep.split(']')
##            nextkeep = secondparentremover[0]
##            realimg = nextkeep.split('+0.j')
##            realpart= float(realimg[0])
##            scientific_notation = "{:.4e}".format(realpart)
##            final = float(scientific_notation)
##
##            2DEigenvectorsIntermediate = np.append(2DEigenvectorsIntermediate, final)
##        zvalues = np.arange(h, 2DEigenvectorsIntermediate.size, ZDIV)
##        zvaluesint = zvalues.astype(int)
##        for val in zvaluesint:
##            2DZArray = np.append(2DZArray, 2DEigenvectorsIntermediate[val])
##        2DZArray = np.reshape(2DZArray, (XDIV, YDIV))
##        ax1.plot_surface(meshx, meshy, 2DZArray)
##        plt.show()        
        
    print(datetime.now()-startTime)

generate1D("comparision", -1.0, 1.0, 3, XLEVEL = 0.0, YLEVEL = 0.0, Overwrite = True)

