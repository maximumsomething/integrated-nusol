import time
import os.path

global timerName
timerName = ""
timerStart = 0
def startTimer(name):
	global timerName
	global timerStart
	if timerName != "":
		endTimer()
	timerName = name
	timerStart = time.process_time()

def endTimer():
	global timerName
	global timerStart
	timerEnd = time.process_time()
	print(timerName, "took", format(timerEnd - timerStart, '.4f'), "s")
	timerName = ""


def checkFileWriteable(path, description, Overwrite):
	if Overwrite:
		try:
			myfile = open(path, 'a')
			myfile.close()
		except IOError:
			print("The", description, "file you are trying to save to is currently open. Close the file(s) and rerun the program again.")
			return False

	elif os.path.exists(path):
		print("The ", description, ' file "', path, '" already exists. Change the name or set Overwrite to True.', sep="")
		return False
	
	return True

def checkFileReadable(path, description):
	if not os.path.exists(path):
		print("The ", description, 'file "', path, '" you are trying to access does not exist.', sep="")
		return False
	else:
		return True

class Filenames:

	def generateinfo(ProjectName, NDIM):
		os.makedirs("generate_info", exist_ok=True)
		return os.path.join("generate_info", "%s%sD_geninfo.dat" %(ProjectName, NDIM))

	def potarray(ProjectName, NDIM):
		os.makedirs("Potential_array", exist_ok=True)
		return os.path.join("Potential_array", "%s%sD_Potential.npy" %(ProjectName, NDIM))

	def valout(ProjectName, NDIM):
		os.makedirs("Eigenvalues_out", exist_ok=True)
		return os.path.join("Eigenvalues_out", "%s%sD_valout.dat" %(ProjectName, NDIM))

	def vecout(ProjectName, NDIM):
		os.makedirs("Eigenevectors_flat", exist_ok=True)
		return os.path.join("Eigenevectors_flat", "%s%sD_vecoutFlat.dat" %(ProjectName, NDIM)) 

	def vecarray(ProjectName, NDIM):
		os.makedirs("Eigenvectors_array", exist_ok=True)
		return os.path.join("Eigenvectors_array", "%s%sD_vecarray.npy" %(ProjectName, NDIM)) 

	def potentialAnalysis(ProjectName, NDIM):
		os.makedirs("potential_analysis", exist_ok=True)
		return os.path.join("potential_analysis", "%s%sD_analysis.txt" %(ProjectName, NDIM))

	




