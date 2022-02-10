import gemmi
import numpy as np

from generate_potential import atom


def atoms_from_cif(file, radius=6.0, BINDING_LABEL="D1", ORIGIN_LABEL="O1", EXCLUDED_SITES=["D1", "D2", "D3", "D4", "D5", "D6"]):


	structure = gemmi.read_small_structure(file)

	print(structure)
	print(structure.cell)
	print("Num sites:", len(structure.sites))
	# print(structure[0])


	for site in structure.sites:
		if site.label == BINDING_LABEL:
			bindingSite = site
		# if site.label == ORIGIN_LABEL:
		# 	originSite = site



	ns = gemmi.NeighborSearch(structure, radius*2).populate()
	print(ns)
	neighbors = ns.find_site_neighbors(bindingSite, max_dist=radius)

	# Search for nearest atom with the origin site's label
	for mark in neighbors:
		site = mark.to_site(structure)
		if site.label == ORIGIN_LABEL:
			originSite = site
			originPos = mark.pos()
			break

	# originPos = originSite.orth(structure.cell)
	bindingPos = bindingSite.orth(structure.cell)


	translationVec = np.array([-originPos.x, -originPos.y, -originPos.z])

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

	print(translationVec)
	print(bindingPos)
	print(rotationMatrix)



	# Lennard-Jones sigma and epsilon values for each element in MOF-5
	LJTable = {
		"Zn": [2.46, 62.4],
		"O": [3.12, 30.2],
		"C": [3.43, 52.8],
		"H": [2.57, 22.1]
	}

	# eliminate duplicate atoms, which gemmi might give you due to symmetry
	trimmedNeighbors = {} # Table of positions and marks
	for mark in neighbors:
		trimmedNeighbors[HashablePoint(mark.x, mark.y, mark.z)] = mark

	print("Num atoms:", len(trimmedNeighbors))


	atoms = []


	for _, mark in trimmedNeighbors.items():
		print(mark)

		# print(mark.x, mark.y, mark.z)
		# print(mark.pos())
		pos = np.array([mark.x, mark.y, mark.z])
		print(pos)
		pos = pos + translationVec
		print(pos)
		pos = np.matmul(rotationMatrix, pos)
		print(pos)
		

		mSite = mark.to_site(structure)
		print(mSite)
		print(mSite.element)

		if mSite.label not in EXCLUDED_SITES:

			LJVals = LJTable[mSite.element.name]

			# Note: the files don't seem to actually have charge
			atoms.append(atom(pos[0], pos[1], pos[2], charge=mSite.charge, sigma=LJVals[0], epsilon=LJVals[1], mass=mSite.element.weight))

		
	return atoms


# Point that can be used in a Dictionary and will compare as equal to very similar points
class HashablePoint:
	def __init__(self, x, y, z):
		self.x = x; self.y = y; self.z = z

	def __hash__(self):
		return int(round(self.x, 1) * 1000000) + int(round(self.y, 1) * 1000) + int(round(self.z, 1))

	def __eq__(self, other):
		return round(self.x, 2) == round(other.x, 2) and round(self.y, 2) == round(other.y, 2) and round(self.z, 2) == round(other.z, 2)




