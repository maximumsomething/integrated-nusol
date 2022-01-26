import numpy as np
import matplotlib.pyplot as plt





XMIN = -1.0
XMAX = 1.0
XDIV = 23
YMIN = -1.0
YMAX = 1.0
YDIV = 23
ZMIN = 3.32
ZMAX = 5.32
ZDIV = 23

XSTEPSIZE = (XMAX-XMIN)/(XDIV-1)

ZArray = np.array([])


s = int((XMAX/XSTEPSIZE))




B = np.array([])

xgrid = np.linspace(XMIN, XMAX, XDIV)
ygrid = np.linspace(YMIN, YMAX, YDIV)
zgrid = np.linspace(ZMIN, ZMAX, ZDIV)
meshx, meshy, meshz = np.meshgrid(xgrid, ygrid, zgrid, sparse=False, indexing="xy")


    

A=(np.load("23by231point0evec.npy"))
print(A)

for val in A:
    string_val = str(val)
    parentremover = string_val.split('[')
    keep = parentremover[1]
    secondparentremover = keep.split(']')
    nextkeep = secondparentremover[0]
    realimg = nextkeep.split('+0.j')
    realpart= float(realimg[0])
    scientific_notation = "{:.4e}".format(realpart)
    final = float(scientific_notation)

    
    B = np.append(B, final)
    
B = np.reshape(B, (XDIV, YDIV, ZDIV))
print(B)
ZArray = B[s:s+1,s:s+1]
ZArray = np.resize(ZArray,(ZDIV))
print(ZArray)
plt.plot(zgrid, ZArray)
plt.show()
