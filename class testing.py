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


XMIN = -1.5
XMAX = 1.5
YMIN = -1.5
YMAX = 1.5
ZMIN = 3.32
ZMAX = 6.32
NDIM = 3
XDIV = 23
YDIV = 23
ZDIV = 23
N_EVAL = 3
hydrogensigma = 2.571
hydrogenepsilon = 0.0000701127
Ck = 8.9875517923E9
alpha = 1
Xgrid = np.linspace(XMIN, XMAX, XDIV)
Ygrid = np.linspace(YMIN, YMAX, YDIV)
Zgrid = np.linspace(ZMIN, ZMAX, ZDIV)

hx = Xgrid[1] - Xgrid[0]
hy = Ygrid[1] - Ygrid[0]






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

def function():
    LJPOL = np.array([])
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
    print(LJPOL)

function()
    
