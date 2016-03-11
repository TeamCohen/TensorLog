# (C) William W. Cohen and Carnegie Mellon University, 2016

import tensorlog

import scipy.sparse as SS
import numpy.random as NR

class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog):
        self.prog = prog
        self.xSyms = []
        self.ySyms = []
        #TODO functions for these
        self.empiricalLoss = None
#       self.regularizationLoss = None

    def addData(mode,filename):
        #TODO allow data for multiple modes, maybe one mode per file
        self.mode = mode
        xs = []
        ys = []
        for line in open(filename):
            sx,sy = line.strip().split("\t")
            xSyms.append(sx)
            ySyms.append(sy)
            xs.append(self.prog.db.onehot(sx))
            ys.append(self.prog.db.onehot(sy))
        self.X = scipy.sparse.vstack(xs)
        self.Y = scipy.sparse.vstack(ys)

    def initializeWeights(self):
        ones = self.prog.db.ones()
        n = self.prog.db.dim()
        initWeights = ones + 0.01*NR.randn(dim)
        self.prog.setWeights(initWeights)

if __name__ == "__main__":
    learner = Learner( tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"]) )

    

        
        
