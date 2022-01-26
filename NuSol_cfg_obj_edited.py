import sys
import numpy as np
import scipy.sparse as sp

class NuSol_cfg_obj():
  """READ CONFIG AND DETERMINE WHAT TO DO ;-)"""
  def __init__(self, cfg):
    self.N_EVAL = int  (cfg.get('OPTIONS','N_EVAL'))
    self.METHOD = str  (cfg.get('OPTIONS','METHOD')).lower()
    self.NDIM   = int  (cfg.get('OPTIONS','NDIM'))
    self.MASS   = float(cfg.get('OPTIONS','MASS'))
    self.HBAR   = float(cfg.get('OPTIONS','HBAR'))
    self.NGRIDX = int  (cfg.get('OPTIONS','NGRIDX'))
    self.XMIN   = float(cfg.get('OPTIONS','XMIN'))
    self.XMAX   = float(cfg.get('OPTIONS','XMAX'))
    self.X = np.linspace(self.XMIN, self.XMAX, self.NGRIDX)
    #grid spacing ... must be equal in all directions
    self.h = self.X[1]-self.X[0]
    self.h = self.X[1]-self.X[0]
    if self.NDIM > 1:
      self.NGRIDY = int(  cfg.get('OPTIONS','NGRIDY'))
      self.YMIN   = float(cfg.get('OPTIONS','YMIN'))
      self.YMAX   = float(cfg.get('OPTIONS','YMAX'))
      self.Y = np.linspace(self.YMIN, self.YMAX, self.NGRIDY)
      self.hY = self.Y[1]-self.Y[0]
      if (self.h != self.hY) and self.METHOD != 'chebyshev':
        print 'gridspacing in X-Y dimension is not equal dx=%f, dy=%f' % (self.h,self.hY)
        print 'correct NGRIDX,NGRIDY in your cfg file'
        print 'exiting... '
        sys.exit()
    if self.NDIM == 3:
      self.NGRIDZ = int(  cfg.get('OPTIONS','NGRIDZ'))
      self.ZMIN   = float( cfg.get('OPTIONS','ZMIN'))
      self.ZMAX   = float( cfg.get('OPTIONS','ZMAX'))
      self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NGRIDZ)
      self.hZ = self.Z[1]-self.Z[0]
      if (self.h != self.hY or self.h != self.hZ or self.hZ != self.hY) and self.METHOD != 'chebyshev':
        print self.h, self.hY, self.hZ, self.METHOD
        print 'ERR: gridspacing in X-Y-Z dimension is not equal dx=%f, dy=%f, dz=%f' % (self.h,self.hY,self.hZ)
        print 'correct NGRIDX,NGRIDY,NGRIDZ in your cfg file'
        print 'exiting... '
        #sys.exit()


    self.USE_FEAST       = cfg.get('OPTIONS','USE_FEAST').lower()
    self.FEAST_PATH      = cfg.get('OPTIONS','FEAST_PATH').strip('\'')
    self.FEAST_MATRIX_OUT_PATH = cfg.get('OPTIONS','FEAST_MATRIX_OUT_PATH')
    self.POTENTIAL_PATH  = cfg.get('OPTIONS','POTENTIAL_PATH')
    self.EIGENVALUES_OUT = cfg.get('OPTIONS','EIGENVALUES_OUT')
    self.EIGENVECTORS_OUT= cfg.get('OPTIONS','EIGENVECTORS_OUT')
    self.FEAST_M         = int(cfg.get('OPTIONS','FEAST_M'))
    self.FEAST_E_MIN     = float(cfg.get('OPTIONS','FEAST_E_MIN'))
    self.FEAST_E_MAX     = float(cfg.get('OPTIONS','FEAST_E_MAX'))

    self.preFactor1D = - 6.0 * self.HBAR * self.HBAR / (self.MASS * self.h * self.h) # prefactor 1D Numerov
    self.preFactor2D =   1.0 / (self.MASS * self.h * self.h / (self.HBAR * self.HBAR))
    self.preFactor3D =  - (self.HBAR * self.HBAR) / (2.0 * self.MASS * self.h * self.h)


    self.USER_FUNCTION       = cfg.get('OPTIONS','USER_FUNCTION')
    self.USE_USER_FUNCTION   = bool(cfg.get('OPTIONS','USE_USER_FUNCTION'))
    if self.USER_FUNCTION != '' and self.USE_USER_FUNCTION:
      print 'evaluation USER_FUNCTION %s' %(self.USER_FUNCTION)
      if self.NDIM == 1:
        x = self.X
        self.V = np.zeros( self.NGRIDX )
        self.V = np.array(eval(self.USER_FUNCTION))
        # np.save(self.POTENTIAL_PATH, self.V)
      if self.NDIM == 2:
        # XY notation for potential... index V[x,y,z]
        x,y = np.meshgrid(self.Y,self.X)
        self.V = np.zeros( (self.NGRIDX, self.NGRIDY) )
        self.V = np.array(eval(self.USER_FUNCTION))
        # np.save(self.POTENTIAL_PATH, self.V)
      if self.NDIM == 3:
        if self.METHOD.find('chebyshev') < 0:
          # XY notation for potential... index V[x,y,z]
          x,y,z = np.meshgrid(self.Y,self.X,self.Z)
          self.V = np.zeros( (self.NGRIDX, self.NGRIDY, self.NGRIDZ) )
          self.V = np.array(eval(self.USER_FUNCTION))
          # np.save(self.POTENTIAL_PATH, self.V)
    else:
      # XY notation for potential... index V[x,y,z]
      print 'loading scanned potential %s' %(self.POTENTIAL_PATH)
      self.V     = np.load(cfg.get('OPTIONS','POTENTIAL_PATH'))

  def MAP_COORDS_REV(self,x_in,x_min,x_max):
    a=x_min
    b=x_max
    x_out = x_in * ((b-a) * 0.5)  + ( (b-a) * 0.5 - np.abs(a) )
    return x_out

  def WRITE_EVAL_AND_EVEC(self,eval,evec):
    norder = eval.argsort()
    eval = eval[norder].real
    evec = evec.T[norder].real
    f = open(self.EIGENVALUES_OUT,'w')
    for e in eval:
      print >> f, "%.12f" % (e)
    f.close()

    f = open(self.EIGENVECTORS_OUT,'w')
    if self.NDIM == 1:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,0,0,self.XMIN,self.XMAX,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    elif self.NDIM == 2:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,self.NGRIDY,0,self.XMIN,self.XMAX,self.YMIN,self.YMAX,0.0,0.0,0.0,0.0,0.0)
    elif self.NDIM == 3:
      print >>f , "%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % (self.NGRIDX,self.NGRIDY,self.NGRIDZ,self.XMIN,self.XMAX,self.YMIN,self.YMAX,self.ZMIN,self.ZMAX,0.0,0.0,0.0)
    for e in evec:
      line=''
      for i in e:
        line+="%.12e " % i
      print >> f, line
    f.close()