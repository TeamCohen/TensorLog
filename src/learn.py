# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog
import time

import numpy as NP
import collections
import mutil

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

#todo is data part of learner?
class Learner(object):
    """Multi-predicate support: initialize Learner with X as a list of matrices, and stack the Ys in a single matrix.
    """

    # prog pts to db, rules
    def __init__(self,prog,X,Y,multiPredicate=False):
        self.prog = prog
        self.X = X
        self.Y = Y
        self.multiPredicate = multiPredicate

    @staticmethod
    def accuracy(Y,P):
        #TODO surely there's a better way of doing this
        n = mutil.numRows(P)
        ok = 0.0
        def allZerosButArgmax(d):
            result = NP.zeros_like(d)
            result[d.argmax()] = 1.0
            return result
        for i in range(n):
            pi = P.getrow(i)
            yi = Y.getrow(i)
            ti = mutil.mapData(allZerosButArgmax,pi)
            ok += yi.multiply(ti).sum()
        return ok/n

    @staticmethod
    def crossEntropy(Y,P):
        """Compute cross entropy some predications relative to some labels."""
        logP = mutil.mapData(NP.log,P)
        return -(Y.multiply(logP).sum())

    def predict(self,mode,X=None):
        """Make predictions on a data matrix associated with the given mode.
        If X==None, use the training data. 
        
        For multiple predicates, returned predications are stacked for use downstream."""
        if X==None: X = self.X
        def impl(m,x):
            predictFun = self.prog.getPredictFunction(m)
            return predictFun.eval(self.prog.db, [x])
        if type(mode) == type(""):
            # then just predict on one mode
            return impl(mode,X)
        else:
            # then mode and X are parallel lists
            init = False
            result = None
            for i in range(len(mode)):
                r = impl(mode[i],X[i])
                if not init: 
                    result = r
                    init = True
                else: result = mutil.stack((result,r))
            return result
    def crossEntropyGrad(self,mode,traceFun=None):
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
        def check(m):
            predictFun = self.prog.getPredictFunction(m)
            assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'
        if not self.multiPredicate: check(mode)
        else:
            for m in mode: check(m)

        X,Y = self.X,self.Y
        P = self.predict(mode,X)
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
#            mean = SS.csr_matrix(delta.mean(axis=0))  #column mean
            m = m0 + mutil.mean(delta)*rate
            #clip negative entries to zero
            m = mutil.mapData(lambda d:NP.clip(m.data,0.0,NP.finfo('float64').max), m)
            self.prog.db.setParameter(functor,arity,m)

class FixedRateGDLearner(Learner):

    def __init__(self,prog,X,Y,epochs=10,rate=0.1,multiPredicate=False):
        super(FixedRateGDLearner,self).__init__(prog,X,Y,multiPredicate)
        self.epochs=epochs
        self.rate=rate
    
    def train(self,mode):
        startTime = time.time()
        for i in range(self.epochs):
            def traceFunForEpoch(thisLearner,Y,P):
                print 'epoch %d of %d' % (i+1,self.epochs),
                print ' crossEnt %.3f' % thisLearner.crossEntropy(Y,P),
                print ' acc %.3f' % thisLearner.accuracy(Y,P),            
                print ' cumSecs %.3f' % (time.time()-startTime)
            paramGrads = self.crossEntropyGrad(mode,traceFun=traceFunForEpoch)
            self.applyMeanUpdate(paramGrads,self.rate)
        

if __name__ == "__main__":
    pass

