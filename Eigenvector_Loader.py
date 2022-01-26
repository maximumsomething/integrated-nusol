import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')



XMIN = -1.75
XMAX = 1.75
XDIV = 23
YMIN = -1.75
YMAX = 1.75
YDIV = 23
ZMIN = 3.32
ZMAX = 7.82
ZDIV = 23


ZSTEPSIZE = (ZMAX-ZMIN)/(ZDIV-1)
print(ZSTEPSIZE)
ZArray = np.array([])

Zstart = 3.32


h = (Zstart - ZMIN)/ZSTEPSIZE
print(h)

B = np.array([])

xgrid = np.linspace(XMIN, XMAX, XDIV)
ygrid = np.linspace(YMIN, YMAX, YDIV)
zgrid = np.linspace(ZMIN, ZMAX, ZDIV)

print(zgrid)
meshx, meshy = np.meshgrid(xgrid, ygrid, sparse=False, indexing="xy")


    

A=(np.load("nuevectest23by231point75.npy"))
#print(A)

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

print(B.size)
zvalues = (np.arange(h, B.size, ZDIV))
zvaluesint = zvalues.astype(int)
print(zvaluesint)

for val in zvaluesint:
    ZArray = np.append(ZArray, B[val])

ZArray = np.reshape(ZArray, (XDIV, YDIV))

B = np.reshape(B, (XDIV, YDIV, ZDIV))

print(B)
print(ZArray)

ax1.plot_surface(meshx, meshy, ZArray)

plt.show()


