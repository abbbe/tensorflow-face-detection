import csv

class Profiler:
    def __init__(self):
        self.f = open('profiler.csv', 'w')
        self.writer = csv. writer(self.f)

    def report(self, t):
        tCapture = t[1] - t[0]
        #tPreProc = t[2] - t[1]
        tFaceProc = t[3] - t[2]
        #tBoxProc = t[4] - t[3]
        tVis = t[5] - t[4]
        tTotal = t[5] - t[1]
        self.writer.writerow([tCapture, tFaceProc, tVis, tTotal])

    def close(self):
        self.writer = None
        self.f.close()
