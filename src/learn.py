# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import time
import numpy as NP
import collections

import tensorlog
import dataset
import mutil

ROBUST_ACCURACY_CHECK = False

class GradAccumulator(object):
    """ Accumulate the sum gradients for perhaps many parameters, indexing
    them by parameter name.
    """
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
        """Increment the parameter with the given name by the appropriate
        amount."""
        if not paramName in self.runningSum:
            self.runningSum[paramName] = deltaGradient
        else:
            self.runningSum[paramName] = self.runningSum[paramName] + deltaGradient

#TODO is data part of learner?
class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog):
        self.prog = prog

    #
    # using and measuring performance
    #

    def predict(self,mode,X):
        """Make predictions on a data matrix associated with the given mode."""
        predictFun = self.prog.getPredictFunction(mode)
        return predictFun.eval(self.prog.db, [X])

    @staticmethod
    def accuracy(Y,P):
        """Evaluate accuracy of predictions P versus labels Y."""
        #TODO surely there's a better way of doing this
        def allZerosButArgmax(d):
            result = NP.zeros_like(d)
            result[d.argmax()] = 1.0
            return result
        n = mutil.numRows(P)
        ok = 0.0
        for i in range(n):
            if not ROBUST_ACCURACY_CHECK:
                pi = P.getrow(i)
                yi = Y.getrow(i)
                ti = mutil.mapData(allZerosButArgmax,pi)
                ok += yi.multiply(ti).sum()
            else:
                #TODO: WAY slower
                maxp = -1
                maxj = -1
                for j in mutil.nzCols(P,i):
                    if P[i,j]>maxp:                
                        maxp = P[i,j]
                        maxj = j
                ok += Y[i,maxj]
        return ok/n

    @staticmethod
    def crossEntropy(Y,P):
        """Compute cross entropy some predications relative to some labels."""
        logP = mutil.mapData(NP.log,P)
        return -(Y.multiply(logP).sum())

    #
    # gradient of crossEntropy loss
    #

    def crossEntropyGrad(self,mode,X,Y,traceFun=None):
        """Compute the parameter gradient associated with softmax
        normalization followed by a cross-entropy cost function.
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
        predictFun = self.prog.getPredictFunction(mode)
        assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'

        P = self.predict(mode,X)
        if traceFun: traceFun(self,Y,P)
        paramGrads = GradAccumulator()
        #TODO assert rowSum(Y) = all ones - that's assumed here in
        #initial delta of Y-P
        predictFun.fun.backprop(Y-P,paramGrads)
        return paramGrads

    #
    # parameter update
    #

    def applyMeanUpdate(self,paramGrads,rate):
        """ Compute the mean of each parameter gradient, and add it to the
        appropriate param, after scaling by rate. If necessary clip
        negative parameters to zero.
        """ 
        for (functor,arity),delta in paramGrads.items():
            m0 = self.prog.db.getParameter(functor,arity)
            m = m0 + mutil.mean(delta)*rate
            #clip negative entries to zero
            NP.clip(m.data,0.0,NP.finfo('float64').max)
            self.prog.db.setParameter(functor,arity,m)

class FixedRateGDLearner(Learner):
    """ Very simple one-predicate learner
    """  

    def __init__(self,prog,epochs=10,rate=0.1):
        super(FixedRateGDLearner,self).__init__(prog)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,mode,X,Y):
        startTime = time.time()
        for i in range(self.epochs):
            def traceFunForEpoch(thisLearner,Y,P):
                print 'epoch %d of %d' % (i+1,self.epochs),
                print ' crossEnt %.3f' % thisLearner.crossEntropy(Y,P),
                print ' acc %.3f' % thisLearner.accuracy(Y,P),            
                print ' cumSecs %.3f' % (time.time()-startTime)
            paramGrads = self.crossEntropyGrad(mode,X,Y,traceFun=traceFunForEpoch)
            self.applyMeanUpdate(paramGrads,self.rate)
        
class MultiPredLearner(Learner):

    def multiPredict(self,dset,copyXs=True):
        """ Return predictions as a dataset. """
        xDict = {}
        yDict = {}
        for mode in dset.modesToLearn():
            X = dset.getX(mode)
            xDict[mode] = X if copyXs else None
            yDict[mode] = self.prog.getPredictFunction(mode).eval(self.prog.db, [X])
        return dataset.Dataset(xDict,yDict)

    @staticmethod
    def multiAccuracy(goldDset,predictedDset):
        weightedSum = 0.0
        totalWeight = 0.0
        for mode in goldDset.modesToLearn():
            assert predictedDset.hasMode(mode)
            Y = goldDset.getY(mode)
            P = predictedDset.getY(mode)
            weight = mutil.numRows(Y)
            weightedSum += weight * Learner.accuracy(Y,P)
            totalWeight += weight
        return weightedSum/totalWeight

    @staticmethod
    def multiCrossEntropy(goldDset,predictedDset):
        result = 0.0
        for mode in goldDset.modesToLearn():
            assert predictedDset.hasMode(mode)
            Y = goldDset.getY(mode)
            P = predictedDset.getY(mode)
            result += Learner.crossEntropy(Y,P)
        return result

class MultiPredFixedRateGDLearner(MultiPredLearner):

    def __init__(self,prog,epochs=10,rate=0.1):
        super(MultiPredFixedRateGDLearner,self).__init__(prog)
        self.epochs=epochs
        self.rate=rate
    
    def multiTrain(self,dset):
        startTime = time.time()
        for i in range(self.epochs):
            for mode in dset.modesToLearn():
                def myTraceFun(thisLearner,Y,P):
                    print 'epoch %d of %d: target mode %s' % (i+1,self.epochs,str(mode)),
                    print ' crossEnt %.3f' % thisLearner.crossEntropy(Y,P),
                    print ' acc %.3f' % thisLearner.accuracy(Y,P),            
                    print ' cumSecs %.3f' % (time.time()-startTime)
                paramGrads = self.crossEntropyGrad(mode,dset.getX(mode),dset.getY(mode),traceFun=myTraceFun)
                self.applyMeanUpdate(paramGrads,self.rate)
            
if __name__ == "__main__":
    pass

