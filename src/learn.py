# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog

import scipy.sparse as SS
import numpy.random as NR
import numpy as NP
import collections

#TODO clean up mode/modeString

class UpdateAccumulator(object):
    def __init__(self):
        self.updates = collections.defaultdict(list) #for debug
        self.runningSum = {}
    def keys(self):
        return self.runningSum.keys()
    def items(self):
        return self.runningSum.items()
    def __getitem__(self,paramName):
        return self.runningSum[paramName]
    def __setitem__(self,paramName,gradient):
        self.runningSum[paramName] = gradient
    def accum(self,paramName,deltaGradient):
        self.updates[paramName].append(deltaGradient)
        if not paramName in self.runningSum:
            self.runningSum[paramName] = deltaGradient
        else:
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

    def applyMeanUpdate(self,updates,rate):
        for (functor,arity),delta in updates.items():
            m0 = self.prog.db.getParameter(functor,arity)
            print 'm0',type(m0),m0
            print 'delta',type(delta),delta
            #TODO - mean returns a dense matrix, can I avoid that
            m = m0 + SS.csr_matrix(delta.mean(axis=1))*rate
            #clip negative entries to zero
            print 'm',type(m),m
            clippedData = NP.clip(m.data,0.0,NP.finfo('float64'))
            m = SS.csr_matrix((clippedData, m.indices, m.indptr), shape=m.shape)
            m.prog.db.setParameter(functor,arity,m)

class FixedRateSGDLearner(Learner):

    def __init__(self,prog,data,epochs=10,rate=0.01):
        super(FixedRateSGDLearner,self).__init__(prog,data)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,modeString):
        for i in range(self.epochs):
            print 'epoch',i+1,'of',self.epochs
            updates = self.crossEntropyUpdate(modeString)
            self.applyMeanUpdate(updates,self.rate)

if __name__ == "__main__":
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
    learner = Learner(prog)
    learner.addData(tensorlog.ModeDeclaration('predict(i,o)'), 'test/textcattoy-train.examples')
    learner.initializeWeights()
    print 'params',learner.prog.db.params
    learner.crossEntropyUpdate()

