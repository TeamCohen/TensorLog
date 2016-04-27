# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog

import scipy.sparse as SS
import numpy.random as NR
import numpy as NP
import collections
import bcast

#TODO clean up mode/modeString

class GradAccumulator(object):
    def __init__(self):
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
        return self.getX(mode),self.getY(mode)
    def getX(self,mode):
        assert self.xs[mode], 'no data inserted for mode %r' % mode
        return SS.vstack(self.xs[mode])
    def getY(self,mode):
        assert self.ys[mode], 'no labels inserted for mode %r' % mode
        return SS.vstack(self.ys[mode])

class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog,data):
        self.prog = prog
        self.data = data

    @staticmethod
    def accuracy(Y,P):
        #TODO surely there's a better way of doing this
        n = bcast.numRows(P)
        ok = 0.0
        for i in range(n):
            pi = P.getrow(i)
            yi = Y.getrow(i)
            di = pi.data
            tdata = NP.zeros_like(di)
            maxj = di.argmax()
            tdata[maxj] = 1
            ti = SS.csr_matrix((tdata, pi.indices, pi.indptr), shape=pi.shape, dtype='float64')
            ok += yi.multiply(ti).sum()
        return ok/n

    @staticmethod
    def crossEntropy(Y,P):
        """Compute cross entropy some predications relative to some labels."""
        logPData = NP.log(P.data)
        logP = SS.csr_matrix((logPData, P.indices, P.indptr), shape=P.shape, dtype='float64')
        return -(Y.multiply(logP).sum())

    def predict(self,modeString,X=None):
        """Make predictions on a data matrix associated with the given mode.
        If X==None, use the training data. """
        if not X:
            X = self.data.getX(modeString)
        mode = tensorlog.ModeDeclaration(modeString)
        predictFun = self.prog.getPredictFunction(mode)
        return predictFun.eval(self.prog.db, [X])

    def crossEntropyGrad(self,modeString,traceFun=None):
        """Compute parameter gradient associated with softmax normalization
        followed by a cross-entropy cost function.
        """

        # More detail: in learning we use a softmax normalization
        # followed immediately by a crossEntropy loss, which has a
        # simple derivative when combined - see
        # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
        # So in doing backprop, you don't call backprop on the outer
        # function, instead you compute the initial delta of P-Y, the
        # derivative for the loss of the (softmax o crossEntropy)
        # function, and it pass that delta down to the inner function
        # for softMax

        # a check
        predictFun = self.prog.getPredictFunction(tensorlog.ModeDeclaration(modeString))
        assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'

        X,Y = self.data.getData(modeString)
        P = self.predict(modeString,X)
        if traceFun: traceFun(self,Y,P)
        paramGrads = GradAccumulator()
        #TODO assert rowSum(Y) = all ones - that's assumed here in
        #initial delta of Y-P
        predictFun.fun.backprop(Y-P,paramGrads)
        return paramGrads

    def applyMeanUpdate(self,paramGrads,rate):
        """ Compute the mean of each parameter gradient, and add it to the
        appropriate param, after scaling by rate. If necessary clip
        negative parameters to zero.
        """ 
        for (functor,arity),delta in paramGrads.items():
            m0 = self.prog.db.getParameter(functor,arity)
            #TODO - mean returns a dense matrix, can I avoid that?
            mean = SS.csr_matrix(delta.mean(axis=0))  #column mean
            m = m0 + mean*rate
            #clip negative entries to zero
            clippedData = NP.clip(m.data,0.0,NP.finfo('float64'))
            m = SS.csr_matrix((clippedData, m.indices, m.indptr), shape=m.shape, dtype='float64')
            self.prog.db.setParameter(functor,arity,m)

class FixedRateGDLearner(Learner):

    def __init__(self,prog,data,epochs=10,rate=0.1):
        super(FixedRateGDLearner,self).__init__(prog,data)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,modeString):
        for i in range(self.epochs):
            def traceFunForEpoch(thisLearner,Y,P):
                print 'epoch %d of %d' % (i+1,self.epochs),
                print ' crossEnt %.3f' % thisLearner.crossEntropy(Y,P),
                print ' acc %.3f' % thisLearner.accuracy(Y,P)            
            paramGrads = self.crossEntropyGrad(modeString,traceFun=traceFunForEpoch)
            self.applyMeanUpdate(paramGrads,self.rate)
        

if __name__ == "__main__":
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
    learner = Learner(prog)
    learner.addData(tensorlog.ModeDeclaration('predict(i,o)'), 'test/textcattoy-train.examples')
    learner.initializeWeights()
    print 'params',learner.prog.db.params
    learner.crossEntropyGrad()

