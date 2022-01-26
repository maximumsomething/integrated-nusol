import copy as copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
class atom:
    def __init__ (self, x, y, z, charge=0, sigma=0, epsilon=0, mass=0):
        self.x=x
        self.y=y
        self.z=z
        self.charge=charge
        self.sigma=sigma
        self.epsilon=epsilon
        self.mass=mass
    def __str__(self):
        return "x is %s, y is %s, z is %s, " \
               "charge is %s, sigma is %s, epsilon is %s, " \
               "mass is %s" % (self.x, self.y, self.z, self.charge,
                               self.sigma, self.epsilon, self.mass)





Zn1 = atom(.29385, .20615, .166667, 0, 0, 0, 0)
O1 = atom(.25, .25, .25, 0, 0, 0, 0)
O2 = atom(.28067, .21934, .13334, 0, 0, 0, 0)
C1 = atom(.25, .25, .11183, 0, 0, 0, 0)
C2 = atom(.25, .25, .05334, 0, 0, 0, 0)
C3 = atom(.28261, .21739, .02665, 0, 0, 0, 0)
H3 = atom(.30869, .19131, .04833, 0, 0, 0, 0)

MOF5atoms =[atom(-1.855325180842072, 0.0, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
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



baseatoms = MOF5atoms


totalatoms = []


for atoms in baseatoms:
    totalatoms.append(atoms)




def flip(xflip, yflip, zflip):
    new_atoms = copy.deepcopy(baseatoms)
    if xflip == "x":
        for atoms in new_atoms:
            atoms.x = atoms.x
    if xflip == "-x":
        for atoms in new_atoms:
            atoms.x = -atoms.x
    if yflip == "y":
        for atoms in new_atoms:
            atoms.y = atoms.y
    if yflip == "-y":
        for atoms in new_atoms:
            atoms.y = -atoms.y
    if zflip == "z":
        for atoms in new_atoms:
            atoms.z = atoms.z
    if zflip == "-z":
        for atoms in new_atoms:
            atoms.z = -atoms.z
    for atoms in new_atoms:
        totalatoms.append(atoms)

def translate(xtrans, ytrans, ztrans):
    new_atoms = copy.deepcopy(baseatoms)
    for atoms in new_atoms:
        atoms.x = atoms.x + xtrans
        atoms.y = atoms.y + ytrans
        atoms.z = atoms.z + ztrans
    for atoms in new_atoms:
        totalatoms.append(atoms)

def flipntranslate(xflip, yflip, zflip, xtrans, ytrans, ztrans):
    new_atoms = copy.deepcopy(baseatoms)
    if xflip == "x":
        for atoms in new_atoms:
            atoms.x = atoms.x
    if xflip == "-x":
        for atoms in new_atoms:
            atoms.x = -atoms.x
    if yflip == "y":
        for atoms in new_atoms:
            atoms.y = atoms.y
    if yflip == "-y":
        for atoms in new_atoms:
            atoms.y = -atoms.y
    if zflip == "z":
        for atoms in new_atoms:
            atoms.z = atoms.z
    if zflip == "-z":
        for atoms in new_atoms:
            atoms.z = -atoms.z
    for atoms in new_atoms:
        atoms.x = atoms.x + xtrans
        atoms.y = atoms.y + ytrans
        atoms.z = atoms.z + ztrans
    for atoms in new_atoms:
        totalatoms.append(atoms)

def translatenflip(xtrans, ytrans, ztrans, xflip, yflip, zflip):
    new_atoms = copy.deepcopy(baseatoms)
    for atoms in new_atoms:
        atoms.x = atoms.x + xtrans
        atoms.y = atoms.y + ytrans
        atoms.z = atoms.z + ztrans
    if xflip == "x":
        for atoms in new_atoms:
            atoms.x = atoms.x
    if xflip == "-x":
        for atoms in new_atoms:
            atoms.x = -atoms.x
    if yflip == "y":
        for atoms in new_atoms:
            atoms.y = atoms.y
    if yflip == "-y":
        for atoms in new_atoms:
            atoms.y = -atoms.y
    if zflip == "z":
        for atoms in new_atoms:
            atoms.z = atoms.z
    if zflip == "-z":
        for atoms in new_atoms:
            atoms.z = -atoms.z
    for atoms in new_atoms:
        totalatoms.append(atoms)

#flip("-x", "-y", "-x")
#flip("-x", "y", "-z")

xs = []
ys = []
zs = []
for atoms in totalatoms:
    xs.append(atoms.x)
    ys.append(atoms.y)
    zs.append(atoms.z)


    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
plt.show()

for atoms in totalatoms:
    print(atoms)
