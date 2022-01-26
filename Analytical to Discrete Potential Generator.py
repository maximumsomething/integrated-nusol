import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

XMIN = -2.0
XMAX = 2.0
XDIV = 27
XSTEPSIZE = (XMAX-XMIN)/(XDIV-1)
print(XSTEPSIZE)
YMIN = -2.0
YMAX = 2.0
YDIV = 27
YSTEPSIZE = (YMAX-YMIN)/(YDIV-1)
ZMIN = -2.0
ZMAX = 2.0
ZDIV = 27
ZSTEPSIZE = (ZMAX-ZMIN)/(ZDIV-1)
xgrid = np.linspace(XMIN, XMAX, XDIV)
ygrid = np.linspace(YMIN, YMAX, YDIV)
zgrid = np.linspace(ZMIN, ZMAX, ZDIV)

B = np.array([])

for xval in xgrid:
    for yval in ygrid:
        for zval in zgrid:
            value = .00042*(xval**2 + yval**2 + zval**2)

            B = np.append(B, value)

B = np.reshape(B, (XDIV, YDIV, ZDIV))
print("Minimum potential is", np.amin(B))

print("The minimum potential's array position is:" , np.unravel_index(np.argmin(B, axis=None), B.shape))

result = (np.where(B == np.amin(B)))



listofcoordinates = list(zip(XMAX-result[0]*XSTEPSIZE, YMAX-result[1]*YSTEPSIZE, ZMAX-result[2]*ZSTEPSIZE))

min_list = []

for coord in listofcoordinates:
    min_list.append(coord)

print("The x,y,z position of the minimum is", (min_list))


minimumpot = np.amin(B)

xresult = result[0]
yresult = result[1]
zresult = result[2]

print(B[xresult-1, yresult, zresult])

print(B[xresult-1, yresult, zresult])

print(XSTEPSIZE*XSTEPSIZE)
#print(B)
xsecondderivative = (B[xresult+1, yresult, zresult] - 2*minimumpot + B[xresult-1, yresult, zresult])/(XSTEPSIZE*XSTEPSIZE)
ysecondderivative = ( 3)
zsecondderivative = (3)
#np.save("newattempttest1.npy", B)

print(xsecondderivative)



            
