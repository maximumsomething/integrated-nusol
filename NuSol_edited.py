#!/usr/bin/python2.7
import numpy as np
import subprocess
import sys
import os
import os.path
from ConfigParser import SafeConfigParser
from NuSol_cfg_obj_edited import NuSol_cfg_obj
from NuSol_matrices_edited import NuSol_matrices
from NuSol_version_checker_edited import NuSol_version
from scipy.linalg import solve
from datetime import datetime
import scipy.optimize as op
import scipy.sparse as sp

startTime = datetime.now()

#def rounder(number):
#    string_number = str(number)
#    parentremover = string_number.split('[')
#    firstparent = parentremover[0]
#    keep = parentremover[1]
#    secondparentremover = keep.split(']')
#    nextkeep = secondparentremover[0]
#    secondparent = secondparentremover[1]
#    realimg = nextkeep.split('+0.j')
#    realpart = float(realimg[0])
#    imgpart = realimg[1]
#    scientific_notation = "{:.4e}".format(realpart)
    
    
#    print(scientific_notation)
  


class numerov():
  def __init__ (self,cfgname):
    cfg = SafeConfigParser()
    cfg.read(cfgname)
    cfg = NuSol_cfg_obj(cfg)
    NuSolM = NuSol_matrices(cfg)

    if cfg.METHOD == 'numerov':
      if cfg.NDIM == 1:
        print ('Creating 1D Numerov Matrix -- %d grid points [X] -- grid spacing %f' % (cfg.NGRIDX,cfg.h))
        A,M = NuSolM.Numerov_Matrix_1D()
      if cfg.NDIM == 2:
        print ('Creating 2D Numerov Matrix -- %dx%d=%d grid points [XY] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDX*cfg.NGRIDY,cfg.h))
        A,M = NuSolM.Numerov_Matrix_2D()
      if cfg.NDIM == 3:
        print ('Creating 3D Numerov Matrix -- %dx%dx%d=%d grid points [XYZ] -- grid spacing %f Bohr' % (cfg.NGRIDX,cfg.NGRIDY,cfg.NGRIDZ,cfg.NGRIDX*cfg.NGRIDY*cfg.NGRIDZ,cfg.h))
        A,M = NuSolM.Numerov_Matrix_3D()
      if cfg.USE_FEAST == 'true' :
        # test if shared libraries for numerov are loaded
        if os.path.exists("%s/NuSol_FEAST"%(cfg.FEAST_PATH)):
          n = subprocess.Popen('ldd %s/NuSol_FEAST| grep "not found" | wc -l'% (cfg.FEAST_PATH),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
          libsloaded = int( n.stdout.readlines()[0].strip('\n') )
          if libsloaded == 0: # run FEAST NUMEROV solver
            p = subprocess.Popen('%s/NuSol_FEAST %f %f %d %s %s %s' % (cfg.FEAST_PATH,cfg.FEAST_E_MIN,cfg.FEAST_E_MAX,cfg.FEAST_M,cfg.FEAST_MATRIX_OUT_PATH,cfg.EIGENVALUES_OUT,cfg.EIGENVECTORS_OUT),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                print (line,)
            retval = p.wait()
          else:
            print ('ERR: Shared libraries for Numerov Feast solver not loaded! Source the intel mkl and check dependencies with:')
            print ('     ldd $PATHTO/NuSol_FEAST')
            sys.exit()
      else:    # run build in ARPACK solver instead
        print ('Note: Using buildin SCIPY ARPACK interface for Numerov.')
        eval,evec = sp.linalg.eigs(A=A,k=cfg.N_EVAL,M=M,which='SM')
        cfg.WRITE_EVAL_AND_EVEC(eval,evec)
        current_dir_path = "C:\Users\Student\Desktop\NuSol_official"
        os.chdir(current_dir_path)
        np.save("outputarray5point5sep23div2.npy", evec)
        #for val in evec:
        #  rounder(val)




if __name__ == "__main__":
  if len(sys.argv) == 2:
    NuV = NuSol_version()
    res = NuV.version_check()
    if res == True:
      if os.path.isfile(sys.argv[1]):
        numerov(sys.argv[1])
      else:
        print ('%s does not seem to exist' % (sys.argv[1]) )
        sys.exit()
    else:
      print ('exiting..')
  else:
    print ('ERR: No config file found! Please provide a config file in the command line:')
    print ('python numerov.py config.cfg')
    sys.exit(1)

print(datetime.now()-startTime)

