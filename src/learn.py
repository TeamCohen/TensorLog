# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog

import scipy.sparse as SS
import numpy.random as NR
import collections

#for backprop updates
def updateAccumulator():
    return collections.defaultdict(list)

class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog):
        self.prog = prog
        self.xSyms = []
        self.ySyms = []
        #TODO functions for these
        self.empiricalLoss = None
#       self.regularizationLoss = None

    def addData(self,mode,filename):
        #TODO allow data for multiple modes, maybe one mode per file
        self.mode = mode
        xs = []
        ys = []
        for line in open(filename):
            sx,sy = line.strip().split("\t")
            self.xSyms.append(sx)
            self.ySyms.append(sy)
            xs.append(self.prog.db.onehot(sx))
            ys.append(self.prog.db.onehot(sy))
        self.X = SS.vstack(xs)
        self.Y = SS.vstack(ys)

    def initializeWeights(self):
        ones = self.prog.db.ones()
        n = self.prog.db.dim()
        initWeights = SS.csc_matrix(ones + 0.01*NR.randn(n))
        self.prog.setWeights(initWeights)

    def showMat(self,msg,m):
        dm = self.prog.db.matrixAsSymbolDict(m)
        for r,d in dm.items():
            print msg,r,d

    def crossEntropyUpdate(self):
        self.predictFun = self.prog.getPredictFunction(self.mode)
        assert isinstance(self.predictFun,funs.SoftmaxFunction),'crossEntropyUpdate specialized to work for softmax normalization'
        #forward propagation
        P = self.predictFun.eval(self.prog.db, [self.X])
        paramUpdates = updateAccumulator()
        self.predictFun.fun.backprop([self.Y - P],paramUpdates)

if __name__ == "__main__":
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
    learner = Learner(prog)
    learner.addData(tensorlog.ModeDeclaration('predict(i,o)'), 'test/textcattoy-train.examples')
    learner.initializeWeights()
    print 'params',learner.prog.db.params
    learner.crossEntropyUpdate()

