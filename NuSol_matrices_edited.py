import operator
import numpy as np
import scipy.sparse as sp


class NuSol_matrices():
  def __init__(self, cfg):
    self.cfg = cfg

  def Numerov_Matrix_1D(self):
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
    NumerovMatrix1D = []
    FORTRANoffset = 1
    Nele = 0
    for i in xrange(self.cfg.NGRIDX):
      NumerovMatrix1D.append(
        [FORTRANoffset + i, FORTRANoffset + i, -2.0 * self.cfg.preFactor1D + 10.0 * self.cfg.V[i], 10.0])
      Nele += 1
      if i - 1 >= 0:
        NumerovMatrix1D.append(
          [FORTRANoffset + i, FORTRANoffset + i - 1, 1.0 * self.cfg.preFactor1D + self.cfg.V[i - 1], 1.0])
        Nele += 1
      if i + 1 < self.cfg.NGRIDX:
        NumerovMatrix1D.append(
          [FORTRANoffset + i, FORTRANoffset + i + 1, 1.0 * self.cfg.preFactor1D + self.cfg.V[i + 1], 1.0])
        Nele += 1

    print   >> f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (
    self.cfg.NGRIDX, Nele, self.cfg.NGRIDX, 0, 0, self.cfg.XMIN, self.cfg.XMAX, 0.0, 0.0, 0.0, 0.0, self.cfg.h, 0.0,
    0.0)
    NumerovMatrix1D = sorted(NumerovMatrix1D, key=operator.itemgetter(0, 1))
    for line in NumerovMatrix1D:
      print   >> f, "%12d%12d % 18.16E % 18.16E" % (line[0], line[1], line[2], line[3])
    f.close()

    NumerovMatrix1D = np.array(NumerovMatrix1D)
    row = NumerovMatrix1D[:, 0] - 1
    col = NumerovMatrix1D[:, 1] - 1
    dataA = NumerovMatrix1D[:, 2]
    dataM = NumerovMatrix1D[:, 3]
    A = sp.coo_matrix((dataA, (row, col)), shape=(self.cfg.NGRIDX, self.cfg.NGRIDX))
    M = sp.csr_matrix((dataM, (row, col)), shape=(self.cfg.NGRIDX, self.cfg.NGRIDX))
    return A, M

  def Numerov_Matrix_2D(self):
    Nx = self.cfg.NGRIDX
    Ny = self.cfg.NGRIDY
    print Nx, Ny
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
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
              [FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1, -   1.0 * self.cfg.preFactor2D, 0.0])
            Nele += 1
          NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
                                  -   4.0 * self.cfg.preFactor2D + self.cfg.V[iN - 1, iK], 1.0])
          Nele += 1
          if (iKy + 1 < Ny):
            NumerovMatrix2D.append(
              [FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1, -   1.0 * self.cfg.preFactor2D, 0.0])
            Nele += 1

        iNx = iN * Ny
        iNy = iN * Ny
        iKx = iK
        iKy = iK
        if (iKy - 1 >= 0):
          NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1,
                                  -  4.0 * self.cfg.preFactor2D + self.cfg.V[iN, iK - 1], 1.0])
          Nele += 1
        NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
                                + 20.0 * self.cfg.preFactor2D + 8.0 * self.cfg.V[iN, iK], 8.0])
        Nele += 1
        if (iKy + 1 < Ny):
          NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1,
                                  -  4.0 * self.cfg.preFactor2D + self.cfg.V[iN, iK + 1], 1.0])
          Nele += 1

        if (iN + 1 < Nx):
          iNx = iN * Ny
          iNy = (iN + 1) * Ny
          iKx = iK
          iKy = iK
          if (iKy - 1 >= 0):
            NumerovMatrix2D.append(
              [FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy - 1, -  1.0 * self.cfg.preFactor2D, 0.0])
            Nele += 1
          NumerovMatrix2D.append([FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy,
                                  -  4.0 * self.cfg.preFactor2D + self.cfg.V[iN + 1, iK], 1.0])
          Nele += 1
          if (iKy + 1 < Ny):
            NumerovMatrix2D.append(
              [FORTRANoffset + iNx + iKx, FORTRANoffset + iNy + iKy + 1, -  1.0 * self.cfg.preFactor2D, 0.0])
            Nele += 1

    print   >> f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (
    self.cfg.NGRIDX * self.cfg.NGRIDY, Nele, self.cfg.NGRIDX, self.cfg.NGRIDY, 0, self.cfg.XMIN, self.cfg.XMAX,
    self.cfg.YMIN, self.cfg.YMAX, 0.0, 0.0, self.cfg.h, self.cfg.h, 0.0)

    NumerovMatrix2D = sorted(NumerovMatrix2D, key=operator.itemgetter(0, 1))
    for line in NumerovMatrix2D:
      print   >> f, "%12d%12d % 18.16E % 18.16E" % (line[0], line[1], line[2], line[3])
    f.close()

    NumerovMatrix2D = np.array(NumerovMatrix2D)
    row = NumerovMatrix2D[:, 0] - 1
    col = NumerovMatrix2D[:, 1] - 1
    dataA = NumerovMatrix2D[:, 2]
    dataM = NumerovMatrix2D[:, 3]
    A = sp.coo_matrix((dataA, (row, col)), shape=(self.cfg.NGRIDX * self.cfg.NGRIDY, self.cfg.NGRIDX * self.cfg.NGRIDY))
    M = sp.csr_matrix((dataM, (row, col)), shape=(self.cfg.NGRIDX * self.cfg.NGRIDY, self.cfg.NGRIDX * self.cfg.NGRIDY))
    return A, M

  def Numerov_Matrix_3D(self):
    Nx = self.cfg.NGRIDX
    Ny = self.cfg.NGRIDY
    Nz = self.cfg.NGRIDZ
    f = open(self.cfg.FEAST_MATRIX_OUT_PATH, 'w')
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
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * self.cfg.preFactor3D, 0.0])
              Nele += 1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1

            iNx = iN * Ny
            iNy = iN * Ny
            iKx = iK
            iKy = iK
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN, iK, iL - 1], 1.0])
            Nele += 1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1

            if (iN + 1 < Nx):
              iNx = iN * Ny
              iNy = (iN + 1) * Ny
              iKx = iK
              iKy = iK
              if (iKy - 1 >= 0):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * self.cfg.preFactor3D, 0.0])
              Nele += 1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * self.cfg.preFactor3D,
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
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN - 1, iK, iL], 1.0])
            Nele += 1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1

          iNx = iN * Ny
          iNy = iN * Ny
          iKx = iK
          iKy = iK
          if (iKy - 1 >= 0):
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN, iK - 1, iL], 1.0])
            Nele += 1

          NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
                                  -72.0 * self.cfg.preFactor3D + 6.0 * self.cfg.V[iN, iK, iL], +6.0])
          Nele += 1

          if (iKy + 1 < Ny):
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN, iK + 1, iL], 1.0])
            Nele += 1

          if (iN + 1 < Nx):
            iNx = iN * Ny
            iNy = (iN + 1) * Ny
            iKx = iK
            iKy = iK
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN + 1, iK, iL], 1.0])
            Nele += 1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * self.cfg.preFactor3D,
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
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * self.cfg.preFactor3D, 0.0])
              Nele += 1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1
            iNx = iN * Ny
            iNy = iN * Ny
            iKx = iK
            iKy = iK
            if (iKy - 1 >= 0):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1
            NumerovMatrix3D.append([FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy,
                                    16.0 * self.cfg.preFactor3D + self.cfg.V[iN, iK, iL + 1], 1.0])
            Nele += 1
            if (iKy + 1 < Ny):
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, -4.0 * self.cfg.preFactor3D,
                 0.0])
              Nele += 1
            if (iN + 1 < Nx):
              iNx = iN * Ny
              iNy = (iN + 1) * Ny
              iKx = iK
              iKy = iK
              if (iKy - 1 >= 0):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy - 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1
              NumerovMatrix3D.append(
                [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy, -4.0 * self.cfg.preFactor3D, 0.0])
              Nele += 1
              if (iKy + 1 < Ny):
                NumerovMatrix3D.append(
                  [FORTRANoffset + iLx + iNx + iKx, FORTRANoffset + iLy + iNy + iKy + 1, 3.0 * self.cfg.preFactor3D,
                   0.0])
                Nele += 1

    print   >> f, "%12d%12d%12d%12d%12d %f %f %f %f %f %f %f %f %f" % (
    self.cfg.NGRIDX * self.cfg.NGRIDY * self.cfg.NGRIDZ, Nele, self.cfg.NGRIDX, self.cfg.NGRIDY, self.cfg.NGRIDZ,
    self.cfg.XMIN, self.cfg.XMAX, self.cfg.YMIN, self.cfg.YMAX, self.cfg.ZMIN, self.cfg.ZMAX, self.cfg.h, self.cfg.h,
    self.cfg.h)
    NumerovMatrix3D = sorted(NumerovMatrix3D, key=operator.itemgetter(0, 1))
    for line in NumerovMatrix3D:
      print   >> f, "%12d%12d % 18.16E % 18.16E" % (line[0], line[1], line[2], line[3])
    f.close()
    NumerovMatrix3D = np.array(NumerovMatrix3D)
    row = NumerovMatrix3D[:, 0] - 1
    col = NumerovMatrix3D[:, 1] - 1
    dataA = NumerovMatrix3D[:, 2]
    dataM = NumerovMatrix3D[:, 3]
    A = sp.coo_matrix((dataA, (row, col)), shape=(
    self.cfg.NGRIDX * self.cfg.NGRIDY * self.cfg.NGRIDZ, self.cfg.NGRIDX * self.cfg.NGRIDY * self.cfg.NGRIDZ))
    M = sp.csr_matrix((dataM, (row, col)), shape=(
    self.cfg.NGRIDX * self.cfg.NGRIDY * self.cfg.NGRIDZ, self.cfg.NGRIDX * self.cfg.NGRIDY * self.cfg.NGRIDZ))
    return A, M
