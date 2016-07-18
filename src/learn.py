# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# learning methods for Tensorlog
# 

import time
import math
import numpy as NP
import scipy.sparse as SS
import collections

import opfunutil
import funs
import tensorlog
import dataset
import mutil
import declare
import traceback
import config

# clip to avoid exploding gradients

conf = config.Config()
conf.minGradient = -100;   conf.help.minGradient = "Clip gradients smaller than this to minGradient"
conf.maxGradient = +100;   conf.help.minGradient = "Clip gradients larger than this to maxGradient"

##############################################################################
# helper classes
##############################################################################

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

class Tracer(object):

    """ Functions to pass in as arguments to a learner's "tracer"
    keyword argument.  These are called by the optimizer after
    gradient computation for each mode - at this point Y and P are
    known.
    """

    @staticmethod
    def silent(learner,Y,P,**kw):
        """No output."""
        pass

    @staticmethod
    def cheap(learner,Y,P,**kw):
        """Easy-to-compute status message."""
        print ' '.join(
            Tracer.identification(learner,kw) 
            + Tracer.timing(learner,kw))
    
    @staticmethod
    def default(learner,Y,P,**kw):
        """A default status message."""
        print ' '.join(
            Tracer.identification(learner,kw) 
            + Tracer.loss(learner,Y,P,kw) 
            + Tracer.timing(learner,kw))

    @staticmethod
    def defaultPlusAcc(learner,Y,P,**kw):
        """A default status message."""
        print ' '.join(
            Tracer.identification(learner,kw) 
            + Tracer.loss(learner,Y,P,kw) 
            + Tracer.accuracy(learner,Y,P,kw) 
            + Tracer.timing(learner,kw))

    #
    # return lists of strings that can be used in a status message,
    # possibly making use of information from the keywords
    # 

    @staticmethod
    def loss(learner,Y,P,kw):
        #perExample=False since we care about the sum xe+reg which is being optimized
        xe = learner.crossEntropy(Y,P,perExample=False)  
        reg = learner.regularizer.regularizationCost(learner.prog)
        return ['loss %.3f' % (xe+reg), 'crossEnt %.3f' % xe, 'reg %.3f' % reg]

    @staticmethod
    def accuracy(learner,Y,P,kw):
        acc = learner.accuracy(Y,P)        
        return ['acc %.3f' % acc]

    @staticmethod
    def timing(learner,kw):
        """Return list of timing properties using keyword 'starttime'
        """
        return ['cumtime %.3f' % (time.time()-kw['startTime'])] if 'startTime' in kw else []

    @staticmethod
    def identification(learner,kw):
        """Return list of identifying properties taken from keywords and learner.
        Known keys are:
           i = current epoch
           k = current minibatch
           mode = current mode
        """
        result = []
        if 'k' in kw: result.append('minibatch %d' % kw['k'])
        if 'i' in kw: result.append('epoch %d of %d' % (kw['i']+1,learner.epochs))
        if 'mode' in kw: result.append('mode %s' % (str(kw['mode'])))
        return result

class EpochTracer(Tracer):

    """Functions to called by a learner after gradient computation for all
    modes and parameter updates.
    """

    @staticmethod
    def silent(learner,dset,**kw):
        """No output."""
        pass

    @staticmethod
    def cheap(learner,dset,**kw):
        """Easy-to-compute status message."""
        print ' '.join(
            EpochTracer.identification(learner,kw) 
            + EpochTracer.timing(learner,kw))
    
    @staticmethod
    def default(learner,dset,**kw):
        """A default status message."""
        P = learner.datasetPredict(dset)
        print ' '.join(
            Tracer.identification(learner,kw) 
            + EpochTracer.epochLoss(learner,dset,P,kw) 
            + Tracer.timing(learner,kw))

    @staticmethod
    def defaultPlusAcc(learner,dset,**kw):
        """A default status message."""
        P = learner.datasetPredict(dset)
        print ' '.join(
            Tracer.identification(learner,kw) 
            + EpochTracer.epochLoss(learner,dset,P,kw) 
            + EpochTracer.epochAccuracy(learner,dset,P,kw) 
            + Tracer.timing(learner,kw))

    @staticmethod
    def epochLoss(learner,dset,P,kw):
        xe = learner.datasetCrossEntropy(dset,P,perExample=False)
        reg = learner.regularizer.regularizationCost(learner.prog)
        return ['loss %.3f' % (xe+reg), 'crossEnt %.3f' % xe, 'reg %.3f' % reg]

    @staticmethod
    def epochAccuracy(learner,dset,P,kw):
        acc = learner.datasetAccuracy(dset,P)        
        return ['acc %.3f' % acc]


##############################################################################
# Learners
##############################################################################


class Learner(object):
    """Abstract class with some utility functions.."""

    # prog pts to db, rules
    def __init__(self,prog,regularizer,tracer,epochTracer):
        self.prog = prog
        self.regularizer = regularizer or NullRegularizer()
        self.tracer = tracer or Tracer.default
        self.epochTracer = epochTracer or EpochTracer.default

    #
    # using and measuring performance
    #

    def predict(self,mode,X,pad=None):
        """Make predictions on a data matrix associated with the given mode."""
        if not pad: pad = opfunutil.Scratchpad() 
        predictFun = self.prog.getPredictFunction(mode)
        result = predictFun.eval(self.prog.db, [X], pad)
        return result

    def datasetPredict(self,dset,copyXs=True):
        """ Return predictions on a dataset. """
        xDict = {}
        yDict = {}
        for mode in dset.modesToLearn():
            X = dset.getX(mode)
            xDict[mode] = X if copyXs else None
            try:
                #yDict[mode] = self.prog.getPredictFunction(mode).eval(self.prog.db, [X])
                yDict[mode] = self.predict(mode,X)
            except FloatingPointError:
                print "Trouble with mode %s" % str(mode)
                raise
        return dataset.Dataset(xDict,yDict)

    @staticmethod
    def datasetAccuracy(goldDset,predictedDset):
        """ Return accuracy on a dataset relative to gold labels. """
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
            result += Learner.crossEntropy(Y,P,perExample=False)/divisor
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

    def crossEntropyGrad(self,mode,X,Y,tracerArgs={},pad=None):
        """Compute the parameter gradient associated with softmax
        normalization followed by a cross-entropy cost function.  If a
        scratchpad is passed in, then intermediate results of the
        gradient computation will be saved on that scratchpad.
        """

        if not pad: pad = opfunutil.Scratchpad()

        # More detail: in learning we use a softmax normalization
        # followed immediately by a crossEntropy loss, which has a
        # simple derivative when combined - see
        # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
        # So in doing backprop, you don't call backprop on the outer
        # function, instead you compute the initial delta of P-Y, the
        # derivative for the loss of the (softmax o crossEntropy)
        # function, and it pass that delta down to the inner function
        # for softMax

        # do the prediction, saving intermediate outputs on the scratchpad
        predictFun = self.prog.getPredictFunction(mode)
        assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'
        P = self.predict(mode,X,pad)

        # output some status information
        self.tracer(self,Y,P,**tracerArgs)

        # compute gradient
        paramGrads = GradAccumulator()
        #TODO assert rowSum(Y) = all ones - that's assumed here in
        #initial delta of Y-P
        predictFun.fun.backprop(Y-P,paramGrads,pad)

        return paramGrads


    #
    # parameter update
    #

    def applyMeanUpdate(self,paramGrads,rate,n,totalN=0):
        """ Compute the mean of each parameter gradient, and add it to the
        appropriate param, after scaling by rate. If necessary clip
        negative parameters to zero.
        """ 

        for (functor,arity),delta0 in paramGrads.items():
            #clip the delta vector to avoid exploding gradients
            delta = mutil.mapData(lambda d:NP.clip(d,conf.minGradient,conf.maxGradient), delta0)

            #figure out how to do the update...
            m0 = self.prog.db.getParameter(functor,arity)
            if mutil.numRows(m0)==1:
                #for a parameter that is a row-vector, we have one
                #gradient per example and we will take the mean
                if totalN>0: 
                    #adjust for a minibatch
                    rate *= float(n)/totalN
                m1 = m0 + mutil.mean(delta)*rate
            else:
                #for a parameter that is a matrix, we have one gradient for the whole matrix
                rateCompensation = (1.0/n) if totalN==0 else (1.0/totalN)
                m1 = m0 + delta*rate*rateCompensation
            #clip negative entries of parameters to zero
            m = mutil.mapData(lambda d:NP.clip(d,0.0,NP.finfo('float64').max), m1)
            #update the parameter
            self.prog.db.setParameter(functor,arity,m)

#
# actual learner implementations
#

class OnePredFixedRateGDLearner(Learner):
    """ Simple one-predicate learner.
    """  
    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,tracer=None,epochTracer=None):
        super(OnePredFixedRateGDLearner,self).__init__(prog,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,mode,X,Y):
        startTime = time.time()
        for i in range(self.epochs):
            n = mutil.numRows(X)
            args = {'i':i,'startTime':startTime}
            paramGrads = self.crossEntropyGrad(mode,X,Y,tracerArgs=args)
            self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
            self.applyMeanUpdate(paramGrads,self.rate,n)

class FixedRateGDLearner(Learner):
    """ A batch gradient descent learner.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,tracer=None,epochTracer=None):
        super(FixedRateGDLearner,self).__init__(prog,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        numModes = len(modes)
        for i in range(self.epochs):
            for j,mode in enumerate(dset.modesToLearn()):
                n = mutil.numRows(dset.getX(mode))
                modeDescription = '%s (%d/%d)' % (str(mode),j+1,numModes)
                args = {'i':i,'startTime':startTime,'mode':modeDescription}
                paramGrads = self.crossEntropyGrad(mode,dset.getX(mode),dset.getY(mode),tracerArgs=args)
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                self.applyMeanUpdate(paramGrads,self.rate,n)
            if numModes>1:
                self.epochTracer(self,dset,i=i,startTime=startTime)
            

class FixedRateSGDLearner(FixedRateGDLearner):

    """ A stochastic gradient descent learner.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,tracer=None,miniBatchSize=100):
        super(FixedRateSGDLearner,self).__init__(prog,epochs=epochs,rate=rate,regularizer=regularizer)
        self.miniBatchSize = miniBatchSize
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        n = len(modes)
        for i in range(self.epochs):
            k = 0
            for (mode,X,Y) in dset.minibatchIterator(batchSize=self.miniBatchSize):
                n = mutil.numRows(X)
                k = k+1
                args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
                paramGrads = self.crossEntropyGrad(mode,X,Y,tracerArgs=args)
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                self.applyMeanUpdate(paramGrads,self.rate,n)

            self.epochTracer(self,dset,i=i,startTime=startTime)

##############################################################################
# regularizers
##############################################################################


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
            # want to do this update, from m->m1
            # m1 = m*(1 - regularizationConstant)^n = m*d
            # but it has to be done addititively so use 
            #     [ m1 = m + z = m*d] 
            # and solve for z which gives you this
            z0 = m * math.pow((1.0 - self.regularizationConstant), n) - m
            z = mutil.repeat(z0,n) if arity==1 else z0
            paramGrads.accum((functor,arity), z)

    def regularizationCost(self,prog):
        result = 0
        for functor,arity in prog.db.params:
            m = prog.db.getParameter(functor,arity)
            result += (m.data * m.data).sum()
        return result*self.regularizationConstant

