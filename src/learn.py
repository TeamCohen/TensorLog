# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog
import time

import numpy as NP
import collections
import mutil
import declare
import logging as L
logging = L.getLogger()

MIN_PROBABILITY = NP.finfo(dtype='float64').eps

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
    def __init__(self,prog,X=None,Y=None):
        self.prog = prog
        self.X = X
        self.Y = Y

    @staticmethod
    def accuracy(Y,P):
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
    def crossEntropy(Y,P):
        """Compute cross entropy some predications relative to some labels."""
        logP = mutil.mapData(NP.log,P)
        return -(Y.multiply(logP).sum())

    def predict(self,mode,X=None):
        """Make predictions on a data matrix associated with the given mode.
        If X==None, use the training data. """
        if X==None: X = self.X
        predictFun = self.prog.getPredictFunction(mode)
        result = predictFun.eval(self.prog.db, [X])
        #mutil.reNormalize(result,threshold=MIN_PROBABILITY)
        return result

    def crossEntropyGrad(self,mode,traceFun=None,X=None,Y=None):
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
        
        if X==None: X=self.X
        if Y==None: Y=self.Y
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
            NP.clip(m.data,0.0,NP.finfo('float64').max)
            self.prog.db.setParameter(functor,arity,m)

class FixedRateGDLearner(Learner):

    def __init__(self,prog,X,Y,epochs=10,rate=0.1):
        super(FixedRateGDLearner,self).__init__(prog,X,Y)
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

class MultiModeLearner(FixedRateGDLearner):

    def __init__(self,prog,modes,Xs=None,Ys=None,data=None,epochs=10,rate=0.1):
        super(MultiModeLearner,self).__init__(prog,X=None,Y=None,epochs=epochs,rate=rate)
        self.modes = modes
        if data!=None: 
            self.trainingData = data
            self.Ys = [data[m.functor][1] for m in modes]
        else:
            self.trainingData = {}
            for (mode,x,y) in zip(modes,Xs,Ys):
                self.trainingData[mode.functor] = (x,y)
            self.Ys = Ys
        self.rate = self.rate / len(self.modes)
    def train(self):
        startTime = time.time()
        batches = len(self.modes)
        for i in range(self.epochs):
            print 'epoch %d of %d' % (i+1,self.epochs)
            lastPrint = time.time()
            for b in range(batches):
                if time.time()-lastPrint > 10: # seconds
                   print 'batch %d of %d: %s...' % (b+1,batches,str(self.modes[b]))
                   lastPrint = time.time()
                if self.modes[b].functor not in self.trainingData: assert "No training data available for mode %s" % str(self.modes[b])
                try:
                    self._trainBatch(self.modes[b],*self.trainingData[self.modes[b].functor],startTime=startTime)
                    if logging.isEnabledFor(L.DEBUG):
                        for p in self.prog.db.params:
                            logging.debug("params in %s: max %g min %g sum %s",str(p),self.prog.db.getParameter(*p).max(),self.prog.db.getParameter(*p).min(),self.prog.db.getParameter(*p).sum())
                except FloatingPointError as e:
                    print "_trainBatch trouble at mode %d, %s" % (b,str(self.modes[b]))
                    raise
            Ps = self.predict(self.modes,data=self.trainingData)
            print ' crossEnt %.3f' % self.crossEntropy(self.Ys,Ps),
            print ' acc %.3f' % self.accuracy(self.Ys,Ps),
            print ' cumSecs %.3f' % (time.time()-startTime)
            
    def _trainBatch(self,mode,X,Y,startTime):
        paramGrads = self.crossEntropyGrad(mode,X=X,Y=Y)
        if logging.isEnabledFor(L.DEBUG):
            for (functor,arity),delta in paramGrads.items():
                logging.debug("paramGrads for %s: max %g min %g %s",functor,delta.max(),delta.min(),mutil.summary(delta))
        self.applyMeanUpdate(paramGrads,self.rate)
                
    def predict(self,modes,Xs=None,data=None):
        if type(modes) == declare.ModeDeclaration:
            return super(MultiModeLearner,self).predict(modes,Xs)
        Y = []
        
        if Xs == None: 
            if data == None: data = self.trainingData
            Xs = [data[m.functor][0] for m in modes]
        i=0
        for m,x in zip(modes,Xs):
            i+=1
            logging.debug("Predict: mode %d of %d: %s %s",i,len(modes),str(m),str(x.shape))
            try:
                Y.append(super(MultiModeLearner,self).predict(m,x))
            except FloatingPointError as e:
                print "predict trouble at mode %d, %s" % (i,str(m))
                raise
        return Y
                             
    def accuracy(self,Ys,Ps,stack=True):
        if stack:
            Ys=mutil.stack(Ys)
            Ps=mutil.stack(Ps)
        return super(MultiModeLearner,self).accuracy(Ys,Ps)
        #acc = []
        #for y,p in zip(Ys,Ps):
        #    acc.append(super(MultiModeLearner,self).accuracy(y,p))
        #return acc
                           
    def crossEntropy(self,Ys,Ps,stack=True):
        if stack:
            Ys=mutil.stack(Ys)
            Ps=mutil.stack(Ps)
        return super(MultiModeLearner,self).crossEntropy(Ys,Ps)
        #xent = []
        #for y,p in zip(Ys,Ps):
        #    xent.append(super(MultiModeLearner,self).crossEntropy(y,p))
        #return xent
    

if __name__ == "__main__":
    pass

