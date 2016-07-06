# (C) William W. Cohen and Carnegie Mellon University, 2016
#

import funs
import time
import math
import numpy as NP
import scipy.sparse as SS
import collections

import tensorlog
import dataset
import mutil
import declare
import traceback

# clip to avoid exploding gradients

MIN_GRADIENT = -100.0
MAX_GRADIENT = +100.0

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
        mutil.checkCSR(deltaGradient,('deltaGradient for %s' % str(paramName)))
        if not paramName in self.runningSum:
            self.runningSum[paramName] = deltaGradient
        else:
            self.runningSum[paramName] = self.runningSum[paramName] + deltaGradient
            mutil.checkCSR(self.runningSum[paramName],('runningSum for %s' % str(paramName)))
        

class Learner(object):
    """Abstract class with some utility functions.."""

    # prog pts to db, rules
    def __init__(self,prog,regularizer,traceFun):
        self.prog = prog
        self.traceFun = traceFun or Learner.defaultTraceFun
        self.regularizer = regularizer or NullRegularizer()

    #
    # using and measuring performance
    #

    def predict(self,mode,X):
        """Make predictions on a data matrix associated with the given mode."""
        predictFun = self.prog.getPredictFunction(mode)
        result = predictFun.eval(self.prog.db, [X])
        return result

    def datasetPredict(self,dset,copyXs=True):
        """ Return predictions on a dataset. """
        xDict = {}
        yDict = {}
        for mode in dset.modesToLearn():
            X = dset.getX(mode)
            xDict[mode] = X if copyXs else None
            try:
                yDict[mode] = self.prog.getPredictFunction(mode).eval(self.prog.db, [X])
            except FloatingPointError:
                print "Trouble with mode %s" % str(mode)
                raise
        return dataset.Dataset(xDict,yDict)

    @staticmethod
    def datasetAccuracy(goldDset,predictedDset):
        """ Return accuracy on a dataset. """
        weightedSum = 0.0
        totalWeight = 0.0
        for mode in goldDset.modesToLearn():
            assert predictedDset.hasMode(mode)
            Y = goldDset.getY(mode)
            P = predictedDset.getY(mode)
            weight = mutil.numRows(Y)
            weightedSum += weight * Learner.accuracy(Y,P)
            totalWeight += weight
        if totalWeight == 0: return 0
        return weightedSum/totalWeight

    @staticmethod
    def datasetCrossEntropy(goldDset,predictedDset,perExample=True):
        """ Return cross entropy on a dataset. """
        result = 0.0
        for mode in goldDset.modesToLearn():
            assert predictedDset.hasMode(mode)
            Y = goldDset.getY(mode)
            P = predictedDset.getY(mode)
            divisor = mutil.numRows(Y) if perExample else 1.0
            result += Learner.crossEntropy(Y,P)/divisor
        return result


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
            pi = P.getrow(i)
            yi = Y.getrow(i)
            ti = mutil.mapData(allZerosButArgmax,pi)
            ok += yi.multiply(ti).sum()
        return ok/n

    @staticmethod
    def crossEntropy(Y,P,perExample=False):
        """Compute cross entropy some predications relative to some labels."""
        logP = mutil.mapData(NP.log,P)
        result = -(Y.multiply(logP).sum())
        return result/mutil.numRows(Y) if perExample else result

    @staticmethod
    def nullTraceFun(thisLearner, Y, P, i = -1, startTime = 0.0):
        """To be passed in as a traceFun option to a Learner.  In each epoch,
        this is called, and does nothing at all.
        """
        pass

    @staticmethod
    def cheapTraceFun(thisLearner, Y, P, i = -1, startTime = 0.0):
        """To be passed in as a traceFun option to a Learner.  In each epoch,
        this is called, and prints some inexpensively-computed statistics.
        """
        print 'epoch %2d of %d' % (i+1,thisLearner.epochs),
        print ' cumSecs %.3f' % (time.time()-startTime)

    @staticmethod
    def defaultTraceFun(thisLearner, Y, P, i = -1, startTime = 0.0, mode=None):
        """To be passed in as a traceFun option to a Learner.  In each epoch,
        this is called, and prints some possibly useful statistics.
        """
        xe = thisLearner.crossEntropy(Y,P)
        reg = thisLearner.regularizer.regularizationCost(thisLearner.prog)
        print 'epoch %2d of %d' % (i+1,thisLearner.epochs),
        if mode: print ' for %s' % mode,
        print ' : loss %.3f = crossEnt %.3f + reg %.3f' % (xe+reg,xe,reg),
        print ' ; acc %.3f' % thisLearner.accuracy(Y,P),
        print ' ; cumSecs %.3f' % (time.time()-startTime)

    def crossEntropyGrad(self,mode,X,Y,traceFunArgs={}):
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

        predictFun = self.prog.getPredictFunction(mode)
        assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'
        P = self.predict(mode,X)
        if self.traceFun: self.traceFun(self,Y,P,**traceFunArgs)
        paramGrads = GradAccumulator()
        #TODO assert rowSum(Y) = all ones - that's assumed here in
        #initial delta of Y-P
        predictFun.fun.backprop(Y-P,paramGrads)
        return paramGrads


    #
    # parameter update
    #

    def applyMeanUpdate(self,paramGrads,rate,n):
        """ Compute the mean of each parameter gradient, and add it to the
        appropriate param, after scaling by rate. If necessary clip
        negative parameters to zero.
        """ 
        for (functor,arity),delta0 in paramGrads.items():
            #clip the delta vector to avoid exploding gradients
            #TODO have a clip function in mutil?
            delta = mutil.mapData(lambda d:NP.clip(d,MIN_GRADIENT,MAX_GRADIENT), delta0)
            m0 = self.prog.db.getParameter(functor,arity)
            if mutil.numRows(m0)==1:
                #for a parameter that is a row-vector, we have one gradient per example
                m1 = m0 + mutil.mean(delta)*rate
            else:
                #for a parameter that is matrix, we have one gradient for the whole
                m1 = m0 + delta*(1.0/n)*rate
            #clip negative entries of parameters to zero
            m = mutil.mapData(lambda d:NP.clip(d,0.0,NP.finfo('float64').max), m1)
            self.prog.db.setParameter(functor,arity,m)

class OnePredFixedRateGDLearner(Learner):
    """ Simple one-predicate learner.
    """  
    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None):
        super(OnePredFixedRateGDLearner,self).__init__(prog,regularizer=regularizer,traceFun=traceFun)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,mode,X,Y):
        startTime = time.time()
        for i in range(self.epochs):
            n = mutil.numRows(X)
            args = {'i':i,'startTime':startTime}
            paramGrads = self.crossEntropyGrad(mode,X,Y,traceFunArgs=args)
            self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
            self.applyMeanUpdate(paramGrads,self.rate,n)

class FixedRateGDLearner(Learner):
    """ A batch gradient descent learner.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None):
        super(FixedRateGDLearner,self).__init__(prog,regularizer=regularizer,traceFun=traceFun)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        n = len(modes)
        for i in range(self.epochs):
            for mode in dset.modesToLearn():
                n = mutil.numRows(dset.getX(mode))
                args = {'i':i,'startTime':startTime,'mode':str(mode)}
                paramGrads = self.crossEntropyGrad(mode,dset.getX(mode),dset.getY(mode),traceFunArgs=args)
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                self.applyMeanUpdate(paramGrads,self.rate,n)
            

class FixedRateSGDLearner(FixedRateGDLearner):

    """ A stochastic gradient descent learner.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None,miniBatchSize=100):
        super(FixedRateSGDLearner,self).__init__(prog,epochs=epochs,rate=rate,regularizer=regularizer)
        self.miniBatchSize = miniBatchSize
        self.traceFun = traceFun or FixedRateSGDLearner.defaultTraceFun
    
    @staticmethod
    def defaultTraceFun(thisLearner, Y, P, i = -1, k = -1, startTime = 0.0, mode = 'unknown'):
        print 'epoch %d of %d miniBatch %d: target mode %s' % (i+1,k+1,thisLearner.epochs,str(mode)),
        print ' crossEnt %.3f' % thisLearner.crossEntropy(Y,P),
        print ' reg cost %.3f' % thisLearner.regularizer.regularizationCost(thisLearner.prog),
        print ' acc %.3f' % thisLearner.accuracy(Y,P),            
        print ' cumSecs %.3f' % (time.time()-startTime)

    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        n = len(modes)
        for i in range(self.epochs):
            k = 0
            for (mode,X,Y) in dset.minibatchIterator(batchSize=self.miniBatchSize):
                try:
                    n = mutil.numRows(X)
                    k = k+1
                    args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
                    paramGrads = self.crossEntropyGrad(mode,X,Y,traceFunArgs=args)
                    self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                    self.applyMeanUpdate(paramGrads,self.rate,n)
                except:
                    print 'error encountered learning on mode %s' % mode
                    print '-' * 60
                    traceback.print_exc()
                    print '-' * 60

class Regularizer(object):
    """Abstract class for regularizers."""

    def addRegularizationGrad(self,paramGrads,prog,n):
        """Introduce the regularization gradient to a GradAccumulator."""
        assert False, 'abstract method called'

    def regularizationCost(self,prog):
        """Report the current regularization cost."""
        assert False, 'abstract method called'

class NullRegularizer(object):
    """ Default case which does no regularization"""

    def addRegularizationGrad(self,paramGrads,prog,n):
        pass

    def regularizationCost(self,prog):
        return 0.0

class L2Regularizer(Regularizer):
    """ L2 regularization toward 0."""

    def __init__(self,regularizationConstant=0.01):
        self.regularizationConstant = regularizationConstant
    
    def addRegularizationGrad(self,paramGrads,prog,n):
        for functor,arity in prog.db.params:
            m = prog.db.getParameter(functor,arity)
            # want to do this update, from m->m1, but addititively
            # m1 = m*(1 - regularizationConstant)^n = m*d
            # so use [ m1 = m + delta = m*d] and solve for delta
            delta0 = m * math.pow((1.0 - self.regularizationConstant), n) - m
            delta = mutil.repeat(delta0,n) if arity==1 else delta0
            paramGrads.accum((functor,arity), delta)

    def regularizationCost(self,prog):
        result = 0
        for functor,arity in prog.db.params:
            m = prog.db.getParameter(functor,arity)
            result += (m.data * m.data).sum()
        return result*self.regularizationConstant

