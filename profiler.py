import csv

# 0-1 capture
# 1-2 move detection (not implemented)
# 2-3 face detection (NN)
# 3-4 face processing
# 4-5 visualization

class Profiler:
    def __init__(self):
        self.f = open('profiler.csv', 'w')
        self.writer = csv. writer(self.f)

    def report(self, t):
        tCapture = t[1] - t[0]
        #tPreProc = t[2] - t[1]
        tFaceDet = t[3] - t[2]
        tFaceProc = t[4] - t[3]
        tVis = t[5] - t[4]
        tTotal = t[5] - t[0]
        self.writer.writerow([tCapture, tFaceDet, tFaceProc, tVis, tTotal])

    def close(self):
        self.writer = None
        self.f.close()
