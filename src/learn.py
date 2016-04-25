# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog

import scipy.sparse as SS
import numpy.random as NR
import collections

class UpdateAccumulator(object):
    def __init__(self):
        self.updates = collections.defaultdict(list) #for debug
        self.runningSum = {}
    def keys(self):
        return self.runningSum.keys()
    def items(self):
        return self.runningSum.items()
    def getUpdate(self,paramName):
        return self.runningSum[paramName]
    def accum(self,paramName,deltaGradient):
        self.updates[paramName].append(deltaGradient)
        if not paramName in self.runningSum:
            self.runningSum[paramName] = deltaGradient
        else:
            #print '!!! paramName increment',paramName,'by',deltaGradient
            self.runningSum[paramName] = self.runningSum[paramName] + deltaGradient

#TODO modes should be objects not strings
class Dataset(object):
    def __init__(self,db):
        self.db = db
        self.xSyms = collections.defaultdict(list)
        self.ySyms = collections.defaultdict(list)
        self.xs = collections.defaultdict(list)
        self.ys = collections.defaultdict(list)
    def addDataSymbols(self,mode,sx,syList):
        """syList is a list of symbols that are correct answers to input sx
        for the function associated with the given mode."""
        assert len(syList)>0, 'need to have some desired outputs for each input'
        self.xSyms[mode].append(sx)
        self.xs[mode].append(self.db.onehot(sx))
        self.ySyms[mode].append(syList)
        distOverYs = self.db.onehot(syList[0])
        for sy in syList[1:]:
            distOverYs = distOverYs + self.db.onehot(sy)
        distOverYs = distOverYs * (1.0/len(syList))
        self.ys[mode].append(distOverYs)
    def getData(self,mode):
        """Return matrix pair X,Y - inputs and corresponding outputs of the
        function for the given mode."""
        assert self.xs[mode], 'no data inserted for mode %r' % mode
        assert self.ys[mode], 'no labels inserted for mode %r' % mode
        return SS.vstack(self.xs[mode]), SS.vstack(self.ys[mode])


class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog,data):
        self.prog = prog
        self.data = data
        #TODO functions for these
        self.empiricalLoss = None
#       self.regularizationLoss = None

#    def initializeWeights(self):
#        ones = self.prog.db.ones()
#        n = self.prog.db.dim()
#        initWeights = SS.csc_matrix(ones + 0.01*NR.randn(n))
#        self.prog.setWeights(initWeights)

    def showMat(self,msg,m):
        dm = self.prog.db.matrixAsSymbolDict(m)
        for r,d in dm.items():
            print msg,r,d

    def crossEntropyUpdate(self,modeString):
        X,Y = self.data.getData(modeString)
        mode = tensorlog.ModeDeclaration(modeString)
        predictFun = self.prog.getPredictFunction(mode)
        assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyUpdate specialized to work for softmax normalization'
        P = predictFun.eval(self.prog.db, [X])
        paramUpdates = UpdateAccumulator()
        delta = Y - P
        predictFun.fun.backprop(delta,paramUpdates)
        return paramUpdates

if __name__ == "__main__":
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
    learner = Learner(prog)
    learner.addData(tensorlog.ModeDeclaration('predict(i,o)'), 'test/textcattoy-train.examples')
    learner.initializeWeights()
    print 'params',learner.prog.db.params
    learner.crossEntropyUpdate()

