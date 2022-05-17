import numpy as np
import sys
import operator
import os
import os.path
from scipy.linalg import solve
import scipy.optimize as op
import scipy.sparse as sp
import scipy.ndimage
import numpy as np
import collections
import ast
np.set_printoptions(threshold=sys.maxsize)

from inus_common_util import *

# A class to store the parameters of each atom used to generate the potential.
# sigma and epsilon are used for Lennard-Jones potential, charge is used for electrostatic potential, 
# coordinates are used for both, and mass is unused.
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
		return f"atom @ ({self.x:.3f}, {self.y:.3f}, {self.z:.3f}) q={self.charge} σ={self.sigma} ε={self.epsilon} m={self.mass}"


# The original atoms from Mof-5 used to generate a test potential.
# MOF5atoms = [atom(-1.855325180842072, 0.0, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
# 		 atom(0.9276625904210358, -1.606758738890191, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
# 		 atom(0.9276625904210358, 1.6067587388901914, 0.656043110880173, 1.8529, 2.4616, 0.0001976046, 0),
# 		 atom(0.0, 0.0, 0.0, -2.2568, 3.118, 0.0000956054, 0),
# 		 atom(-2.2071535575638297, -1.575575329839865, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(2.468065039999284, -1.1236633859835425, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(-0.26091148243545437, 2.6992387158234075, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(2.468065039999284, 1.1236633859835425, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(-0.2609114824354546, -2.6992387158234075, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(-2.2071535575638292, 1.575575329839865, 1.745098923652563, -1.0069, 3.118, 0.0000956054, 0),
# 		 atom(-1.46152887986063, -2.53144227664784, 2.0669139636988607, 1.0982, 3.431, 0.000167333, 0),
# 		 atom(2.92305775972126, 0.0, 2.0669139636988607, -0.1378, 3.431, 0.000167333, 0),
# 		 atom(-1.4615288798606296, 2.53144227664784, 2.0669139636988607, -0.0518, 3.431, 0.000167333, 0)]

# Convert epsilon from hartree energy to kelvin. Uncomment if the above list of atoms are used.
#for changeAtom in MOF5atoms:
#	changeAtom.epsilon *= 315775.3268


# An expanded list of Mof-5 atoms for a test potential, covering one single, complete molecule.
MOF5atoms = [
atom(1.606763689, -0.927665448, 0.655958529, 1.8529, 2.4616, 62.3993, 0),
atom(0.0, 0.0, 0.0, -2.2568, 3.118, 30.19, 0),
atom(1.123635314, 2.468097822, 1.74498432, -1.0069, 3.118, 30.19, 0),
atom(0, 2.923045267, 2.06690513, 1.0982, 3.431, 52.84, 0),
atom(0, 4.160426158, 2.941865549, -0.1378, 3.431, 52.84, 0),
atom(1.194904536, 4.725064489, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(2.150535026, 4.266414844, 3.016810868, .1489, 2.571, 22.14, 0),
atom(-1.123635314, 2.467886268, 1.745283503, -1.0069, 3.118, 30.19, 0),
atom(-1.194904536, 4.725064489, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(-2.150535026, 4.266414844, 3.016810868, .1489, 2.571, 22.14, 0),
atom(1.123635314, 8.109615384, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(0, 7.654667939, 5.412667607, 1.0982, 3.431, 52.84, 0),
atom(0, 6.417287048, 4.537707188, -0.1378, 3.431, 52.84, 0),
atom(1.194904536, 5.852648717, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(2.150535026, 6.311298361, 4.462761869, .1489, 2.571, 22.14, 0),
atom(-1.123635314, 8.109615384, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(-1.194904536, 5.852648717, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(-2.150535026, 6.311298361, 4.462761869, .1489, 2.571, 22.14, 0),
atom(-1.606763689, 0.927665448, 0.655958529, 1.8529, 2.4616, 62.3993, 0),
atom(-2.69925307, -0.260952185, 1.74498432, -1.0069, 3.118, 30.19, 0),
atom(-2.531431458, -1.461522634, 2.06690513, 1.0982, 3.431, 52.84, 0),
atom(-3.603034744, -2.080213079, 2.941865549, -0.1378, 3.431, 52.84, 0),
atom(-4.68947815, -1.327714562, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(-4.770091151, -0.270789458, 3.016810868, .1489, 2.571, 22.14, 0),
atom(-1.575434545, -2.20703986, 1.745283503, -1.0069, 3.118, 30.19, 0),
atom(-3.494573614, -3.397349927, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(-2.619556125, -3.995625386, 3.016810868, .1489, 2.571, 22.14, 0),
atom(-7.584950594, -3.081710965, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(-6.629136892, -3.827333969, 5.412667607, 1.0982, 3.431, 52.84, 0),
atom(-5.557533607, -3.208643524, 4.537707188, -0.1378, 3.431, 52.84, 0),
atom(-5.665994736, -1.891506675, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(-6.541012225, -1.293231217, 4.462761869, .1489, 2.571, 22.14, 0),
atom(-6.46131528, -5.027904418, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(-4.4710902, -3.961142041, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(-4.390477199, -5.018067145, 4.462761869, .1489, 2.571, 22.14, 0),
atom(0, -1.855330896, 0.655958529, 1.8529, 2.4616, 62.3993, 0),
atom(1.575617756, -2.207145638, 1.74498432, -1.0069, 3.118, 30.19, 0),
atom(2.531431458, -1.461522634, 2.06690513, 3.431, 52.84, 0),
atom(3.603034744, -2.080213079, 2.941865549, 3.431, 52.84, 0),
atom(3.494573614, -3.397349927, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(2.619556125, -3.995625386, 3.016810868, .1489, 2.571, 22.14, 0),
atom(2.699069859, -0.260846408, 1.745283503, -1.0069, 3.118, 30.19, 0),
atom(4.68947815, -1.327714562, 3.341125142, -.0518, 3.431, 52.84, 0),
atom(4.770091151, -0.270789458, 3.016810868, .1489, 2.571, 22.14, 0),
atom(6.46131528, -5.027904418, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(6.629136892, -3.827333969, 5.412667607, 3.431, 52.84, 0),
atom(5.557533607, -3.208643524, 4.537707188, -0.1378, 3.431, 52.84, 0),
atom(4.4710902, -3.961142041, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(4.390477199, -5.018067145, 4.462761869, .1489, 2.571, 22.14, 0),
atom(7.584950594, -3.081710965, 5.734588418, -1.0069, 3.118, 30.19, 0),
atom(5.665994736, -1.891506675, 4.138447596, -.0518, 3.431, 52.84, 0),
atom(6.541012225, -1.293231217, 4.462761869, .1489, 2.571, 22.14, 0),
atom(1.606763689, 9.650047758, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(-1.606763689, 9.650047758, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(-9.16056835, -3.433525707, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(9.16056835, -3.433525707, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(7.553804662, -6.216522051, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(-7.553804662, -6.216522051, 6.823614208, 1.8529, 2.4616, 62.3993, 0),
atom(0, 0, -1.967875587, 1.8529, 2.4616, 62.3993, 0),
atom(0, 10.57771321, 7.479572737, -2.2568, 3.118, 30.19, 0),
atom(2.69925307, 9.019510274, 8.307262256, -1.0069, 3.118, 30.19, 0),
atom(-2.699069859, 1.558520264, -0.827689519, -1.0069, 3.118, 30.19, 0),
atom(-0.000183211, -3.116723196, -0.827689519, -1.0069, 3.118, 30.19, 0),
atom(2.699069859, 10.83855961, 5.734289235, -1.0069, 3.118, 30.19, 0),
atom(2.69925307, 1.558202932, -0.827689519, -1.0069, 3.118, 30.19, 0)]

# Lennard-Jones parameters for the hydrogen molecule that will be interacting with the MOF.
hydrogensigma = 2.571
hydrogenepsilon = 22.1398

# Coulumb's constant in K-Å/(e-)^2 for electrostatic potential
Ck = 167101.002

# Polarization constant of hydrogen in Å^3 for electrostatic potential
alpha = 0.675

# This import needs to be here to avoid an infinite loop
from atoms_from_cif import AtomsFromCif

# This class stores all the parameters needed to generate the potential.
class GridInfo:

	def __init__(self, NDIM, XMIN=0.0, XMAX=0.0, XDIV=1, XLEVEL = 0.0, YMIN=0.0, YMAX=0.0, YDIV=1, YLEVEL = 0.0, ZMIN=0.0, ZMAX=0.0, ZDIV=1, ZLEVEL = 0.0, External=True, Estatic=True, CifAtoms=False, AtomSrc=None, Analytic = False, UserFunction = "", Limited = True, PotentialLimit = 10000, axis = None):
		"""
		Construct a GridInfo object.

		The first several parameters specify the location and size of the window.
		NDIM is the number of dimensions (must be 1, 2, or 3).
		XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX specify the position of the window (in angstroms) in 3D space.
		XLEVEL, YLEVEL, ZLEVEL specify the location of the window when NDIM < 3.
		In two dimensions, ZMIN and ZMAX are ignored, and ZLEVEL specifies the z-coordinate of the window.
		In one dimension, axis is either 'x', 'y', or 'z' (defaults to 'z'). 
		For example, if axis='z', ZMIN, ZMAX, XLEVEL, and YLEVEL are used, and XMIN, XMAX, YMIN, YMAX are ignored.
		XDIV, YDIV, ZDIV specify the number of grid points along each dimension. 
		
		The remaining parameters specify details about how the potential is generated:
		If External=True, then no potential is generated. The potential is instead provided to NuSol directly by the caller.
		If Estatic=True, then the electrostatic potential is added to the potential. If Estatic=False, then only the Lennard-Jones potential is used.
		If CifAtoms=True, then a AtomsFromCif object must be provided in AtomSrc. 
		If Analytic=True, then UserFunction must be a string containg a Python expression providing the potential at a point as a function of x, y, and z.
		If Limited=True, then the potential will be capped at PotentialLimit.
		"""

		# We need to check here because the load function might give None instead of letting it default
		if XDIV == None: XDIV = 1
		if YDIV == None: YDIV = 1
		if ZDIV == None: ZDIV = 1

		if axis == None and NDIM == 1: axis = "z"

		if type(NDIM) != int or NDIM>3 or NDIM<1:
			raise ValueError("Number of diminsions must be in an integer format and be no greater than 3 and no less than 1.")
		elif (NDIM == 2 or NDIM == 3) and (type(XMIN)!= float or type(YMIN)!= float or type(XMAX)!= float or type(YMAX)!= float or type(XDIV)!= int or type(YDIV)!= int or XMIN>=XMAX or YMIN>=YMAX or XDIV<= 0 or YDIV<=0):
			raise ValueError("XMIN, XMAX, YMIN, YMAX must all be floats and subject XMIN=<XMAX and YMIN=<YMAX. XDIV, YDIV must be integers greater than zero.")
		elif (NDIM == 1 or NDIM ==3) and (type(ZMIN)!= float or type(ZMAX)!= float or type(ZDIV)!= int or ZMIN>=ZMAX or ZDIV<=0):
			raise ValueError("ZMIN, ZMAX must be floats and subject to ZMIN=<ZMAX. ZDIV must be an integer greater than zero.")
		elif NDIM == 1 and (type(XLEVEL)!= float or type(YLEVEL)!= float):
			raise ValueError("XLEVEL, YLEVEL must be floats.")
		elif NDIM ==2 and type(ZLEVEL)!= float:
			raise ValueError("ZLEVEL must be a float.")
		elif type(Analytic) != bool:
			raise ValueError("Analytic is not in a boolean format. Make sure it is either true or false.")
		elif type(UserFunction) != str and Analytic == True:
			raise ValueError("Function is not in a string format. Make sure the function is in quotation marks and contains only approproiate characters.")
		elif CifAtoms and type(AtomSrc) != AtomsFromCif:
			raise ValueError("AtomSrc must be an AtomsFromCif object.")

		self.NDIM = NDIM

		self.XMIN = XMIN
		self.XMAX = XMAX
		self.XLEVEL = XLEVEL
		self.XDIV = XDIV
		self.YMIN = YMIN
		self.YMAX = YMAX
		self.YLEVEL = YLEVEL
		self.YDIV = YDIV
		self.ZMIN = ZMIN
		self.ZMAX = ZMAX
		self.ZLEVEL = ZLEVEL
		self.ZDIV = ZDIV
		self.External = External
		self.Estatic = Estatic
		self.CifAtoms = CifAtoms
		self.AtomSrc = AtomSrc
		self.Analytic = Analytic
		self.UserFunction = UserFunction
		self.Limited = Limited
		self.PotentialLimit = PotentialLimit
		self.axis = axis

		if NDIM == 1:
			if axis == "z":
				self.XMIN = XLEVEL
				self.XMAX = XLEVEL
				self.YMIN = YLEVEL
				self.YMAX = YLEVEL
			elif axis == "x":
				self.YMIN = YLEVEL
				self.YMAX = YLEVEL
				self.ZMIN = ZLEVEL
				self.ZMAX = ZLEVEL
			elif axis == "y":
				self.ZMIN = ZLEVEL
				self.ZMAX = ZLEVEL
				self.XMIN = XLEVEL
				self.XMAX = XLEVEL
			else:
				raise ValueError("axis must be x, y, or z")

		if NDIM == 2:
			self.ZMIN = ZLEVEL
			self.ZMAX = ZLEVEL

		if Estatic and not (External or Analytic or CifAtoms):
			print("WARNING: Estatic potential gives bad results with sample MOF5atoms")

		self.loadedFromFile = None
		self.warned = False

	def WindowAround3DPoint(SIZE, X=0.0, Y=0.0, Z=0.0, DIV=25, Estatic=True, CifAtoms=False, AtomSrc=None, Analytic=False, UserFunction="", Limited = True, PotentialLimit = 10000.0):
		"""
		This is a convenience function to construct a cubic window. It will create a cube of SIZE angstroms and DIV divisions around 
		the point (X, Y, Z). For a description of the other parameters, see __init__()
		"""

		if type(SIZE) != float:
			raise ValueError("SIZE must be floating-point.")

		return GridInfo(3, X - SIZE/2.0, X + SIZE/2.0, DIV, 0.0, Y - SIZE/2.0, Y + SIZE/2.0, DIV, 0.0, Z - SIZE/2.0, Z + SIZE/2.0, DIV, 0.0, Estatic, CifAtoms, AtomSrc, Analytic, UserFunction, Limited, PotentialLimit)


	def hxyz(self):
		"""
		Returns the grid spacing in each dimension--usually denoted as hx, hy, and hz--of this GridInfo object.
		"""

		try:
			hx = (self.XMAX - self.XMIN) / (self.XDIV - 1)
		except ZeroDivisionError:
			hx = 0.0
		try:
			hy = (self.YMAX - self.YMIN) / (self.YDIV - 1)
		except ZeroDivisionError:
			hy = 0.0
		try:
			hz = (self.ZMAX - self.ZMIN) / (self.ZDIV - 1)
		except ZeroDivisionError:
			hz = 0.0


		if not self.warned:
			self.warned = True
			if self.NDIM == 2 and not inexactEqual(hx, hy):
				print("WARNING: hx and hy must be equal for NuSol to work, but instead, hx=",hx," and hy=",hy, sep="")
			elif self.NDIM == 3 and not (inexactEqual(hx, hy) and inexactEqual(hy, hz)):
				print("WARNING: hx, hy, and hz must be equal for NuSol to work, but instead, hx=",hx,", hy=",hy," and hz=",hz, sep="")

		return hx, hy, hz

	def atoms(self):
		"""
		Returns the atoms used to generate the potential.
		"""
		if self.CifAtoms:
			return self.AtomSrc.atoms
		else: return MOF5atoms

	def saveToFile(self, ProjectName, file):
		"""
		Saves all the parameters of this GridInfo to a file.
		"""
		print("ProjectName=%s\nNDIM=%d\nXMIN=%.8f\nXMAX=%.8f\nXDIV=%d\nXLEVEL=%.8f\nYMIN=%.8f\nYMAX=%.8f\nYDIV=%d\nYLEVEL=%.8f\nZMIN=%.8f\nZMAX=%.8f\nZDIV=%d\nZLEVEL=%.8f\nExternal=%s\nEstatic=%s\nCifAtoms=%s\nAnalytic=%s\nUserFunction=%s\n Limited=%s\n PotentialLimit=%d\n Axis=%s\n" % (ProjectName, self.NDIM, self.XMIN, self.XMAX, self.XDIV, self.XLEVEL, self.YMIN, self.YMAX, self.YDIV, self.YLEVEL, self.ZMIN, self.ZMAX, self.ZDIV, self.ZLEVEL, self.External, self.Estatic, self.CifAtoms, self.Analytic, self.UserFunction, self.Limited, self.PotentialLimit, self.axis), file=file)

		if self.CifAtoms:
			print(f"AtomSrcFile={self.AtomSrc.file}\nAtomSrcRadius={self.AtomSrc.radius}\nBINDING_LABEL={self.AtomSrc.BINDING_LABEL}\nORIGIN_LABEL={self.AtomSrc.ORIGIN_LABEL}\nEXCLUDED_SITES={self.AtomSrc.EXCLUDED_SITES}", file=file)


	def save(self, ProjectName):
		"""
		Saves all the parameters of this GridInfo to the default filename for ProjectName.
		"""
		filename = Filenames.generateinfo(ProjectName, self.NDIM)
		if filename != self.loadedFromFile: # Don't overwrite file that we loaded from
			try:
				f = open(filename, 'w')
				self.saveToFile(ProjectName, f)
					
			except IOError:
				print("WARNING: The potential file", filename, "did not save. The file you wanted to save to was already opened.")

	def load(ProjectName, NDIM):
		"""
		Loads a previously saved GridInfo from the default filename determined by ProjectName and NDIM.

		The file format is flexible. Each line is of the form key=value, where key is the name of the argument to GridInfo's constructor. Unrecognized arguments are ignored. It also supports comments starting with #.
		"""
		return GridInfo.loadFromFile(Filenames.generateinfo(ProjectName, NDIM))


	def loadFromFile(path):
		""" Loads a previously saved GridInfo from the given filename. """
		try:
			file = open(path, "r")
			lines = file.readlines()
			# The first step of the parser is to load every key-value pair into the dictionary v, 
			# which returns None instead of throwing an error when a missing key is accessed.
			v = collections.defaultdict(lambda: None, {})
			linecount = 0
			for line in lines:
				linecount += 1
				line = line.strip() # Remove leading and trailing spaces
				if len(line) > 0 and not line.startswith("#"): # Ignore empty lines and comments
					splitted = line.split('=', 2) # Split into key and value around the equals sign
					if len(splitted) < 2: print("Syntax error loading line ", linecount, ': "', line, '"', sep='')
					v[splitted[0].strip()] = tryParseStringToType(splitted[1].strip())

			print("read from file:", v)

			# Construct the CifAtoms sub-object if necessary
			if v['CifAtoms']:
				AtomSrc = AtomsFromCif(v['AtomSrcFile'], v['AtomSrcRadius'], v['BINDING_LABEL'], v['ORIGIN_LABEL'], v['EXCLUDED_SITES'])
			else: AtomSrc = None

			# Call the constructor with the relevant items of the dictionary
			gridInfo = GridInfo(v['NDIM'], v['XMIN'], v['XMAX'], v['XDIV'], v['XLEVEL'], v['YMIN'], v['YMAX'], v['YDIV'], v['YLEVEL'], v['ZMIN'], v['ZMAX'], v['ZDIV'], v['ZLEVEL'], v['External'], v['Estatic'], v['CifAtoms'], AtomSrc, v['Analytic'], v['UserFunction'], v['Limited'], v['PotentialLimit'], v['Axis'])
			gridInfo.loadedFromFile = path
			return gridInfo
		except IOError:
			print("Error: Could not read potential file", path)
			sys.exit()

# Helper function for load().
# Returns a number, boolean, or array if the input string looks like one of those, or the string otherwise.
def tryParseStringToType(string):
	try:
		return int(string)
	except ValueError:
		try:
			return float(string)
		except ValueError:
			if string == "": return ""
			elif string == "True": return True
			elif string == "False": return False
			if string[0] == '[': 
				# Parse array
				return ast.literal_eval(string)
			else: return string

# Floating point can't exactly represent all numbers, so this function is used to check "equality"
# of floating point numbers from different sources.
def inexactEqual(a, b):
	largest = np.maximum(np.abs(a), np.abs(b))
	return np.abs(a - b) < 0.000000000000001 * largest



def generate(ProjectName, gridInfo, Overwrite = False, PrintAnalysis = True):
	"""
	Generates the potential given the gridInfo object.
	PrintAnalysis: Whether to save the potential and gridInfo to a file, as well as whether to run some analysis on the generate potential.
	Overwrite: Whether to overwrite existing potential, generateinfo, and analysis files.
	"""

	g = gridInfo # for shortness
		#-------4.1-------#
	if type(Overwrite) != bool:
		print("Overwrite is not in a boolean format. Make sure it is either true or false.")
		sys.exit()
	#-------4.2-------#

	print("Generating Potential...")
	startTimer("generate")

	if PrintAnalysis:
		# Get file paths and check whether they are writeable.
		PotentialArrayPath = Filenames.potarray(ProjectName, g.NDIM)
		GenerateInfofile = Filenames.generateinfo(ProjectName, g.NDIM)
		PotentialAnalysisPath = Filenames.potentialAnalysis(ProjectName, g.NDIM)

		checkSuccess = checkFileWriteable(PotentialArrayPath, "Potential Array", Overwrite)
		checkSuccess &= checkFileWriteable(GenerateInfofile, "Generate Info", Overwrite)
		checkSuccess &= checkFileWriteable(PotentialAnalysisPath, "Potential Analysis", Overwrite)

		if not checkSuccess:
			sys.exit()

	if g.Analytic == False:
		# Generate using LJ and maybe electrostatic.
		x, y, z = meshgrids(g)
		V = pointPotential(g.atoms(), g.Estatic, x, y, z)
			       
	elif g.Analytic == True:
		V = generateFromUserFn(g)

	# The above return 3D arrays. Now remove extra dimensions
	if g.NDIM == 1:
		V = np.reshape(V, V.size)
	if g.NDIM == 2:
		V = V[:, :, 0]

	if g.Limited:
		# V = smoothPot(chopPot(V, g.PotentialLimit), g.NDIM)
		# V = smoothChop(V, g.PotentialLimit)
		V = chopPot(V, g.PotentialLimit)
				

	print("########################### \n Done generating potential! \n###########################)")
	#print(V)
	
	if PrintAnalysis == True:
		potAnalFile = open(PotentialAnalysisPath, "w")
		potentialAnalysis(g, potAnalFile, V)

		
		print("########################### \nSaving the potential array as", PotentialArrayPath, "\n###########################")
		np.save(PotentialArrayPath, V)
		
		g.save(ProjectName)
	
	return V	
	
# Generate by eval()-ing the user function.
def generateFromUserFn(g):
	
	x, y, z = meshgrids(g)

	try:
		V = np.array(eval(g.UserFunction))
		# print(V)
	except NameError:
		raise ValueError("Invalid function. Make sure your function is a function of x, y, and z and that all non-elementary operations are proceded by 'np.'")
		sys.exit()

	return V

# Returns the meshgrid for g. The variable x returned by this function is a NDIM-dimensional array with the value at some (integer) coordinate being the x-value (in angstroms) at that coordinate.
def meshgrids(g):
	if g.NDIM == 1 and g.axis == "z":
		Zgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)	
		Xgrid = g.XLEVEL
		Ygrid = g.YLEVEL
	if g.NDIM == 1 and g.axis == "x":
		Xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		Zgrid = g.ZLEVEL
		Ygrid = g.YLEVEL

	if g.NDIM == 1 and g.axis == "y":
		Ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		Xgrid = g.XLEVEL
		Zgrid = g.ZLEVEL

	if g.NDIM == 2:
		Xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		Ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		Zgrid = g.ZLEVEL
	
	if g.NDIM == 3:
		Xgrid = np.linspace(g.XMIN, g.XMAX, g.XDIV)
		Ygrid = np.linspace(g.YMIN, g.YMAX, g.YDIV)
		Zgrid = np.linspace(g.ZMIN, g.ZMAX, g.ZDIV)

	x,y,z = np.meshgrid(Xgrid, Ygrid, Zgrid)
	return (x, y, z)


# Cuts off all potentials greater than VMAX
def chopPot(V, VMAX):
	return np.where(V < VMAX, V, VMAX)


# A test of a few different functions that attempt to smooth the "corner" when the potential is cut off at VMAX. 
# In all three cases, VMAX acts as a "soft" cap where values above it get gradually shrunk.
def smoothChop(V, VMAX):

	# Logarithmic cutoff.
	b = 1 + 1/VMAX
	transformedV = VMAX + np.log(V - VMAX + 1/np.log(b))/np.log(b) - np.log(1/np.log(b))/np.log(b)
	return np.where(V < VMAX, V, transformedV)

	# For the next two smoothing functions, VCHOP represents the "hard" cap, where the potential will not go above no matter what.
	# VCHOP = VMAX * 10.0 

	# parabolic cutoff.
	# transformedV = VMAX + (V - VMAX) * (VCHOP*2 - (V - VMAX))/(VCHOP*2)

	# Exponential cutoff.
	# m = VCHOP * np.exp(1.0)
	# transformedV = VMAX + (V - VMAX) * (np.exp(-(V - VMAX)/m))

	# return np.where(V < VCHOP, np.where(V < VMAX, V, transformedV), VCHOP)

	

# Smoothing test, averages the point with its 2/8/26 neighbors
def smoothPot(V, NDIM):
	if NDIM == 1: kernel = np.ones((3))
	if NDIM == 2: kernel = np.ones((3, 3))
	if NDIM == 3: kernel = np.ones((3, 3, 3))
	kernel /= np.sum(kernel)
	return scipy.ndimage.convolve(V, kernel)


# Prints some important things about the potential, like the minimum, maximum, and second derivative at the minimum.
# Prints both to stdout and the given file.
def potentialAnalysis(g, file, V):
	hx, hy, hz = g.hxyz()


	if not np.isnan(np.sum(V)) and not np.isinf(np.sum(V)):
		# Should we also be doing analysis if there are nans?

		doubleprint(file, "Maximum potential:", np.amax(V), "\nMinimum potential:", np.amin(V), "\nMinimum potential's array position", np.unravel_index(np.argmin(V, axis=None), V.shape))
	
		#result = (np.where(V == np.amin(V)))
		result = np.unravel_index(np.argmin(V), np.shape(V))
		
#-------4.6-------#
		#FIX for axes
		if g.NDIM == 1:
			coord = g.ZMAX-result[0]*hz
			doubleprint(file, coord)
			#doubleprint(file, "The z position of the minimum is", (min_list))
			minimumpot = np.amin(V)
		
			zresult = result[0]
			try:
				zsecondderivative = ((V[zresult+1] - 2*minimumpot + V[zresult-1])/(hz**2))
				doubleprint(file, "The second partial derivative with respect to z is", zsecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to z.")
				zsecondderivative = float("Nan")
			ysecondderivative = float("Nan")
			xsecondderivative = float("Nan")
		if g.NDIM == 2:
			#listofcoordinates = list(zip(g.XMAX-result[0]*hx, g.YMAX-result[1]*hy))
		
			#for coord in listofcoordinates:
			#    min_list.append(coord)

			#doubleprint(file, "The x,y position of the minimum is", (min_list))

			minimumpot = np.amin(V)

			xresult = result[0]
			yresult = result[1]

			try:
				xsecondderivative = ((V[xresult+1, yresult] - 2*minimumpot + V[xresult-1, yresult])/(hx**2))
				doubleprint(file, "The second partial derivative with respect to x is", xsecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to x.")
				xsecondderivative = float("Nan")
			try:
				ysecondderivative = ((V[xresult, yresult+1] - 2*minimumpot + V[xresult, yresult-1])/(hy**2))
				doubleprint(file, "The second partial derivative with respect to y is", ysecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to y.")
				ysecondderivative = float("Nan")
			try: 
				doubleprint(file, "Del Squared is", xsecondderivative+ysecondderivative)
				delsquared = xsecondderivative+ysecondderivative
			except:
				doubleprint(file, "Del Squared is undefined.")
				delsquared = float("Nan")
			zsecondderivative = float("Nan")
		if g.NDIM == 3:
			listofcoordinates = list((g.XMIN+result[0]*hx, g.YMIN+result[1]*hy, g.ZMIN+result[2]*hz))
			doubleprint(file, listofcoordinates)
			#for coord in listofcoordinates:
			#    min_list.append(coord)
			#doubleprint(file, "The x,y,z position of the minimum is", (min_list))
			minimumpot = np.amin(V)
			xresult = result[0]
			yresult = result[1]
			zresult = result[2]
			doubleprint(file, xresult, yresult, zresult)
			try:
				xsecondderivative = ((V[xresult+1, yresult, zresult] - 2*minimumpot + V[xresult-1, yresult, zresult])/(hx**2))
				doubleprint(file, "The second partial derivative with respect to x is", xsecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to x.")
				xsecondderivative = float("nan")
			try:
				ysecondderivative = ((V[xresult, yresult+1, zresult] - 2*minimumpot + V[xresult, yresult-1, zresult])/(hy**2))
				doubleprint(file, "The second partial derivative with respect to y is", ysecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to y.")
				ysecondderivative = float("nan")
			try:
				zsecondderivative = ((V[xresult, yresult, zresult+1] - 2*minimumpot + V[xresult, yresult, zresult-1])/(hz**2))
				doubleprint(file, "The second partial derivative with respect to z is", zsecondderivative)
			except:
				doubleprint(file, "Undefined second partial derivative with respect to z.")
				zsecondderivative = float("nan")
			try: 
				doubleprint(file, "Del Squared is", xsecondderivative+ysecondderivative+zsecondderivative)
				delsquared = xsecondderivative+ysecondderivative+zsecondderivative
			except:
				doubleprint(file, "Del Squared is undefined.")
				delsquared = float("nan")

# Helper function for printAnalysis().
# Prints to both the file and stdout.
def doubleprint(file, *args, **kwargs):
	print(*args, **kwargs)
	print(*args, **kwargs, file=file)


# Calculates the numeric potential for a single point or a meshgrid. 
# atoms is an array of atom, Estatic is a boolean of whether to include the electrostatic potential.
def pointPotential(atoms, Estatic, x, y, z):
	V = LJPotential(atoms, x, y, z)
	if Estatic:
		V += EstaticPotential(atoms, x, y, z)
	return excludeAtoms(V, atoms, x, y, z)

# Calculate the lennard-jones potential for a single point or meshgrid
def LJPotential(atoms, xval, yval, zval):
	LJ = 0
	for atom in atoms:
		jointsigma = (atom.sigma + hydrogensigma)/2
		jointepsilon = np.sqrt(atom.epsilon * hydrogenepsilon)
		magnitude = np.sqrt((xval-atom.x)**2+(yval-atom.y)**2+(zval-atom.z)**2)
		LJpointval = 4*jointepsilon*((jointsigma/magnitude)**12-(jointsigma/magnitude)**6)
		LJ += LJpointval

	return LJ

# Calculate the electrostatic potential for a single point or meshgrid
def EstaticPotential(atoms, x, y, z):
	# E is the electric field at the point,
	# rHat is the unit vector pointing from the atom towards the point,
	# Rsquared is the square of the distance between the atom and the point,
	# atom.charge is the partial charge of the atom,
	# alpha is the polarization constant of the hydrogen molecule, and 
	# Ck is coloumb's constant.

	if type(x) == np.ndarray or type(y) == np.ndarray or type(z) == np.ndarray:
		assert(x.shape == y.shape and x.shape == z.shape)

		E = (np.zeros(x.shape), np.zeros(x.shape), np.zeros(x.shape))
	else:
		E = (0.0, 0.0, 0.0)


	for atom in atoms:
		Rsquared = (x-atom.x)**2+(y-atom.y)**2+(z-atom.z)**2
		R = np.sqrt(Rsquared)
		rHat = (x-atom.x, y-atom.y, z-atom.z) / R
		# print("rHat:", rHat, "R:", R)
		E += atom.charge * rHat / Rsquared

	# print("E:", E)
	Ex, Ey, Ez = E
	return -Ck * (alpha/2) * (Ex**2 + Ey**2 + Ez**2)


# Exclude points within 1A of any atom
def excludeAtoms(V, atoms, x, y, z):
	mask = np.isnan(V)
	for atom in atoms:
		radius = np.sqrt((x-atom.x)**2+(y-atom.y)**2+(z-atom.z)**2)
		mask = mask & (radius < 1.0)

	return np.where(mask, 1000000000, V)


	