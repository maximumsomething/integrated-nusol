import time

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