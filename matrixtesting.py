import operator
from scipy.linalg import solve
import scipy.optimize as op
import scipy.sparse as sp
import numpy as np
N_EVAL = 2
HBAR = 1.0
hz = 1.0
V = np.array([4, 1, 0, 1, 4])
ZDIV = 5
MASS = 1.0

preFactor1D = -6.0* HBAR * HBAR / (MASS * hz * hz)
NumerovMatrix1D = []
for i in range(ZDIV):
    print(i)
    #i step
    NumerovMatrix1D.append(
        [1 + i, 1 + i, -2.0 * preFactor1D + 10.0 * V[i], 10.0])
    print(NumerovMatrix1D)
    #i-1 step if possible
    if i - 1 >= 0:
        NumerovMatrix1D.append(
            [1 + i, i, 1.0 * preFactor1D + V[i - 1], 1.0])
        print("Here is i-1 instance!")
        print(NumerovMatrix1D)
    #i+1 step if possible
    if i + 1 < ZDIV:
        NumerovMatrix1D.append(
            [1 + i, 2 + i, 1.0 * preFactor1D + V[i + 1], 1.0])
        print("Here is i+1 instance!")
        print(NumerovMatrix1D)
NumerovMatrix1D = sorted(NumerovMatrix1D, key=operator.itemgetter(0, 1))
NumerovMatrix1D = np.array(NumerovMatrix1D)
row = NumerovMatrix1D[:, 0] - 1
col = NumerovMatrix1D[:, 1] - 1
dataA = NumerovMatrix1D[:, 2]
dataM = NumerovMatrix1D[:, 3]
A = sp.coo_matrix((dataA, (row, col)), shape=(ZDIV, ZDIV))
M = sp.csr_matrix((dataM, (row, col)), shape=(ZDIV, ZDIV))

eval, evec = sp.linalg.eigs(A=A, k=N_EVAL, M=M, which='SM')
norder = eval.argsort()
eval = eval[norder].real
evec = evec.T[norder].real
print(eval)
