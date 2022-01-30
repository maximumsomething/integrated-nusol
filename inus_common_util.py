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