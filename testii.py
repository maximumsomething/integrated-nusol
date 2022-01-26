import numpy as np
import operator
import scipy.sparse as sp

NumerovMatrix1D = []
XDIV = 4
V = np.array([.5,1,1.5,2])
Nele = 0

print xrange(XDIV)
for i in xrange(XDIV):
    print (i)
for i in xrange(XDIV):
    NumerovMatrix1D.append([1+i, 1+i, V[i], 10.0])
    #print(NumerovMatrix1D)
    Nele += 1
    if i-1 >=0:
        NumerovMatrix1D.append([1+i, 0, V[i-1], 1])
        Nele += 1
        #print(NumerovMatrix1D)
    if i+1 <XDIV:
        NumerovMatrix1D.append([1+i, 2+i, V[i+1], 1])
        Nele +=1
        #print(NumerovMatrix1D)


NumerovMatrix1D = sorted(NumerovMatrix1D, key=operator.itemgetter(0,1))
print(NumerovMatrix1D)

NumerovMatrix1D = np.array(NumerovMatrix1D)
print(NumerovMatrix1D)

row = NumerovMatrix1D[:, 0] - 1
print(row)
col = NumerovMatrix1D[:, 1] - 1
print(col)

dataM = NumerovMatrix1D[:,2]
print(dataM)
dataA = NumerovMatrix1D[:,3]
print(dataA)

A = sp.coo_matrix((dataA, (row, col)), shape=(XDIV, XDIV))
M = sp.csr_matrix((dataM, (row, col)), shape=(XDIV, XDIV))


