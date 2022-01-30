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
	timerStart = time.clock()

def endTimer():
	global timerName
	global timerStart
	timerEnd = time.clock()
	print(timerName, " took ", timerEnd - timerStart, "s")
	timerName = ""