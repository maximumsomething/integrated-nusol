import numpy as np
import scipy.interpolate as interp

from Integrate_Nu_Sol_Prime import numerov
from generate_potential import GridInfo


def interp_uneven_csv(csvfile):

	DFT_V_csv = np.genfromtxt(csvfile, delimiter='	')
	#print(DFT_V_csv)
	interpolator = interp.interp1d(DFT_V_csv[:,0], DFT_V_csv[:,1],'cubic')
	#print(DFT_V_csv[:,0])
	#print(DFT_V_csv[:,1])

	minval = float(np.amin(DFT_V_csv[:,0]))
	maxval = float(np.amax(DFT_V_csv[:,0]))

	diffs = DFT_V_csv[0:-1,0] - DFT_V_csv[1:,0]
	mindiff = np.min(np.abs(diffs))
	
	print("min:", minval, "max:", maxval, "step:", mindiff)

	newx = np.arange(minval, maxval, mindiff)
	newy = interpolator(newx)
	

	numerov("First-DFT", GridInfo(1, ZMIN=minval,ZMAX=maxval,ZDIV=newy.size, External=True), Generate=False, GivenPot=newy, N_EVAL=20, Overwrite=True, IgnoreM=False)