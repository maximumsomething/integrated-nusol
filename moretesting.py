import numpy as np
def ZEigenvectors(ProjectName, NDIM, X, Y, Eval=2):
    print("Starting 1D Eigenvector Analysis!")
    myfile = open("inputs%s3D.dat" %ProjectName, "r")
    lines = myfile.readlines()
    for line in lines:
        line = line.strip()
        line = line.split()
    NDIM = (int(float((line[10]))))
    XMIN = (float(line[3]))
    XMAX = (float(line[4]))
    XDIV = (int(line[0]))
    YMIN = (float(line[5]))
    YMAX = (float(line[6]))
    YDIV = (int(float((line[2]))))
    ZMIN = (float(line[7]))
    ZMAX = (float(line[8]))
    ZDIV = (int(float((line[2]))))
    
    ZArray = np.array([])
    Evecs = np.load("Vecanalysis%s%sD.npy" %(ProjectName, NDIM))
    print(Evecs)
    Evecs = Evecs[Eval-1:Eval]
    print(Evecs)
    Evecs = np.reshape(Evecs,(XDIV, YDIV, ZDIV))
    print(Evecs)

ZEigenvectors("sample", 3, 0, 0, 2)

