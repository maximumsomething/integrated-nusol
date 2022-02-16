import gemmi
import numpy as np
import typing

from inus_common_util import *

from generate_potential import atom


class AtomsFromCif:
	"""
	Loads a CIF (crystallographic information) file, finds all atoms in a radius around the binding site, 
	transforms it so that the origin site is at the origin and the binding site is along the +z axis, and 
	puts it into generate_potential's atom class.

	Currently, it uses a table of lennard-jones and charge parameters for mof-5. In the future, we should expand
	this to something generalizable across different mofs.
	"""

	def __init__(self, file, radius=6.0, BINDING_LABEL="D1", ORIGIN_LABEL="O1", EXCLUDED_SITES=["D1", "D2", "D3", "D4", "D5"]):
		"""
		file: The path to load from
		radius: The radius (in angstroms) around the binding site to include atoms from
		BINDING_LABEL: The label of the binding site in the CIF file
		ORIGIN_LABEL: The label of the atom which will be transformed to be the origin
		EXCLUDED_SITES: Labels of sites we want to exclude from the final list of atoms (usually the binding sites)
		"""


		self.file = file
		self.radius = radius
		self.BINDING_LABEL = BINDING_LABEL
		self.ORIGIN_LABEL = ORIGIN_LABEL
		self.EXCLUDED_SITES = EXCLUDED_SITES

		self.atoms = load_atoms_from_cif(file, radius, BINDING_LABEL, ORIGIN_LABEL, EXCLUDED_SITES)

def load_atoms_from_cif(file, radius=6.0, BINDING_LABEL="D1", ORIGIN_LABEL="O1", EXCLUDED_SITES=["D1", "D2", "D3", "D4", "D5"]):
	if type(file) != str:
		raise ValueError("File path must be a string")
	elif type(radius) != float:
		raise ValueError("Radius must be floating-point")
	elif type(BINDING_LABEL) != str or type(ORIGIN_LABEL) != str:
		raise ValueError("Site labels must be strings")
	elif type(EXCLUDED_SITES) != list:
		raise ValueError("EXCLUDED_SITES must be a list of strings")


	startTimer("Find nearby atoms")

	# Read the file into a gemmi "small structure".
	# As far as I can tell, small structures are appropriate for everything except polymers.
	structure = gemmi.read_small_structure(file)

	print(structure)
	# print(structure.cell)
	print("Num sites:", len(structure.sites))
	# print(structure[0])

	# Find the binding site in the structure
	for site in structure.sites:
		if site.label == BINDING_LABEL:
			bindingSite = site
		# if site.label == ORIGIN_LABEL:
		# 	originSite = site

	bindingPos = bindingSite.orth(structure.cell)


	# Initialize a neighbor search object, with unit cells included up to radius*2.
	# For more information, see https://gemmi.readthedocs.io/en/latest/analysis.html#neighbor-search
	# and https://project-gemmi.github.io/python-api/gemmi.NeighborSearch.html
	ns = gemmi.NeighborSearch(structure, radius*2).populate()
	# print(ns)
	# Find the neighbors within the radius.
	neighbors = ns.find_site_neighbors(bindingSite, max_dist=radius)

	# Search for nearest atom to the binding site with the origin site's label
	originSite = None
	originPos = None
	for mark in neighbors:
		site = mark.to_site(structure)
		if site.label == ORIGIN_LABEL:
			if originSite == None or originPos.dist(bindingPos) > mark.pos().dist(bindingPos):
				originSite = site
				originPos = mark.pos()

	if originSite == None:
		raise ValueError(f"Origin site {ORIGIN_LABEL} not found within radius")
	

	# Get the translation vector which will transform every atom's coordinates so that the origin is at the center.
	translationVec = np.array([-originPos.x, -originPos.y, -originPos.z])

	# Get the rotation matrix which will rotate the binding site onto the +z axis.

	# Get axis vector to become z-axis
	axis = bindingPos - originPos
	# Normalize
	axis /= axis.dist(gemmi.Position(0, 0, 0))
	# z-axis
	targetAxis = gemmi.Position(0, 0, 1)
	# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
	# Cross product is the axis around which the rotation takes place
	cross = axis.cross(targetAxis)
	dot = axis.dot(targetAxis)
	vx = np.array([[0, -cross.z, cross.y], [cross.z, 0, -cross.x], [-cross.y, cross.x, 0]])
	mult = 1/(1 + dot)
	# I + vx + vx^2(1/(1 + dot))
	rotationMatrix = np.array([[1,0,0], [0,1,0], [0,0,1]]) + vx + np.matmul(vx, vx) * mult

	# print(translationVec)
	# print(bindingPos)
	# print(rotationMatrix)



	# Lennard-Jones sigma and epsilon values for each element in MOF-5.
	# In the future, this should be generalized for more MOFs.
	LJTable = {
		"Zn": [2.46, 62.4],
		"O": [3.12, 30.2],
		"C": [3.43, 52.8],
		"H": [2.57, 22.1]
	}

	# Charges for each label in MOF-5
	chargeTable = {
		"Zn1": 1.85,
		"O1": -2.26,
		"O2": -1.01,
		"C1": 1.10,
		"C2": -0.14,
		"C3": -0.05,
		"H3": 0.15,
	}

	# eliminate duplicate atoms at the same position, which gemmi might give you due to symmetry.
	trimmedNeighbors = {} # Table of positions and marks
	for mark in neighbors:
		trimmedNeighbors[HashablePoint(mark.x, mark.y, mark.z)] = mark

	print("Num atoms:", len(trimmedNeighbors))
	

	# Finally, calculate the transformed position of each atom and put it into generate_potential's atom class.
	atoms = []
	for _, mark in trimmedNeighbors.items():
		# print(mark)

		# print(mark.x, mark.y, mark.z)
		# print(mark.pos())
		pos = np.array([mark.x, mark.y, mark.z])
		# print(pos)
		pos = pos + translationVec
		# print(pos)
		pos = np.matmul(rotationMatrix, pos)
		# print(pos)
		

		mSite = mark.to_site(structure)
		# print(mSite)
		# print(mSite.element)

		if mSite.label not in EXCLUDED_SITES:

			LJVals = LJTable[mSite.element.name]

			# Note: the files don't seem to actually have charge
			# charge = mSite.charge
			charge = chargeTable[mSite.label]
			atoms.append(atom(pos[0], pos[1], pos[2], charge=charge, sigma=LJVals[0], epsilon=LJVals[1], mass=mSite.element.weight))

	endTimer()
	return atoms


# Point that can be used in a Dictionary and will compare as equal to very similar points
# Designed for angstrom values in the range 0-100 and looks two places past the decimal.
class HashablePoint:
	def __init__(self, x, y, z):
		self.x = x; self.y = y; self.z = z

	def __hash__(self):
		return int(round(self.x, 1) * 1000000) + int(round(self.y, 1) * 1000) + int(round(self.z, 1))

	def __eq__(self, other):
		return round(self.x, 2) == round(other.x, 2) and round(self.y, 2) == round(other.y, 2) and round(self.z, 2) == round(other.z, 2)




