# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# parallel learning implementation(s)
#

import os
import time
import collections
import multiprocessing
import multiprocessing.pool
import logging
import numpy as NP

from tensorlog import mutil
from tensorlog import learn
from tensorlog import dataset

##############################################################################
# These functions are defined at the top-level of a module so that
# they can be sent to worker processes via pickling.
##############################################################################

def _initWorker(learnerClass,*args):
    """This is called when each subprocess in the ppol is created.
    Learners point to programs which point to DB's so they are large
    objects, so we don't want to include a learner in a task spec;
    instead we provide information that allows each worker to create
    its own learner, which is saved in a global variable called
    'workerLearner'.  Note: this global variable is only defined and
    used for worker subprocesses.
    """
    global workerLearner
    workerLearner = learnerClass(*args)

def _doBackpropTask(task):
    """ Use the workerLearner
    """ 
    (mode,X,Y,args) = task
    paramGrads = workerLearner.crossEntropyGrad(mode,X,Y,tracerArgs=args)
    return (mutil.numRows(X),paramGrads)

def _doAcceptNewParams(paramDict):
    for (functor,arity),value in list(paramDict.items()):
        workerLearner.prog.db.setParameter(functor,arity,value)

def _doPredict(miniBatch):
    (mode,X,Y) = miniBatch
    return (mode,X,workerLearner.predict(mode,X))

##############################################################################
# A parallel learner.
##############################################################################

class ParallelFixedRateGDLearner(learn.FixedRateSGDLearner):
    """Split task into fixed-size miniBatchs and compute gradients of
    these in parallel.  Parameter updates are done only at the end of
    an epoch so this is actually gradient descent, not SGD.
    
    At startup, a pool of workers which share COPIES of the program
    are created, so changes to prog made after the learner is
    initialized are NOT propagated out to the workers.

    parallel is an integer number of workers or None, which will be
    interpreted as the number of CPUs.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,tracer=None,
                 miniBatchSize=100,parallel=10,epochTracer=None):
        tracer = tracer or learn.Tracer.recordDefaults
        super(ParallelFixedRateGDLearner,self).__init__(
            prog,epochs=epochs,rate=rate,regularizer=regularizer,
            miniBatchSize=miniBatchSize,tracer=tracer)
        self.epochTracer = epochTracer or learn.EpochTracer.default
        self.parallel = parallel or multiprocessing.cpu_count()
        logging.info('pool initialized with %d processes' % self.parallel)
        #initargs are used to build a worker learner for the pool,
        #which just computes the gradients and does nothing else
        self.pool = multiprocessing.pool.Pool(
            self.parallel, 
            initializer=_initWorker, 
            #crucial to get the argument order right here!
            initargs=(learn.FixedRateSGDLearner,self.prog,self.epochs,
                      self.rate,self.regularizer,self.tracer,self.miniBatchSize))
        logging.info('created pool of %d workers' % parallel)
    
    #
    # override the learner method with a parallel approach
    #
    def datasetPredict(self,dset,copyXs=True):
        """ Return predictions on a dataset. """
        xDictBuffer = collections.defaultdict(list)
        yDictBuffer = collections.defaultdict(list)
        miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize,shuffleFirst=False))
        logging.info('predicting for %d miniBatches with the worker pool...' % len(miniBatches))
        predictOutputs = self.pool.map(_doPredict, miniBatches, chunksize=1)
        for (mode,X,P) in predictOutputs:
            if copyXs: xDictBuffer[mode].append(X) 
            yDictBuffer[mode].append(P)
        logging.info('predictions for %d miniBatches done' % len(miniBatches))
        xDict = {}
        yDict = {}
        if copyXs:
            for mode in xDictBuffer: 
                xDict[mode] = mutil.stack(xDictBuffer[mode])
        for mode in yDictBuffer: 
            yDict[mode] = mutil.stack(yDictBuffer[mode])
        logging.info('predictions restacked')
        return dataset.Dataset(xDict,yDict)

    @staticmethod
    def miniBatchToTask(batch,i,k,startTime):
        """Convert a minibatch to a task to submit to _doBackpropTask"""
        (mode,X,Y) = batch
        args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
        return (mode,X,Y,args)
        
    def totalNumExamples(self,miniBatches):
        """The total nummber of examples in all the miniBatches"""
        return sum(mutil.numRows(X) for (mode,X,Y) in miniBatches)

    def processGradients(self,bpOutputs,totalN):
        """ Use the gradients to update parameters """
        self.regularizer.regularizeParams(self.prog,totalN)
        for (n,paramGrads) in bpOutputs:
            # scale rate down to reflect the fraction of the data  
            self.applyUpdate(paramGrads, self.rate * (float(n)/totalN))

    def broadcastParameters(self):
        """" Broadcast the new parameters to the subprocesses """
        paramDict = dict(
            ((functor,arity),self.prog.db.getParameter(functor,arity))
            for (functor,arity) in self.prog.db.paramList)
        #send one _doAcceptNewParams task to each worker. warning:
        # this seems to work fine....but it is not guaranteed to
        # work from the API, but
        self.pool.map(_doAcceptNewParams, [paramDict]*self.parallel, chunksize=1)
        
    #
    # basic learning routine
    # 

    def train(self,dset):
        modes = dset.modesToLearn()
        trainStartTime = time.time()
        for i in range(self.epochs):
            logging.info("starting epoch %d" % i)
            startTime = time.time()
            #generate the tasks
            miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize))
            bpInputs = [ParallelFixedRateGDLearner.miniBatchToTask(k_b[1],i,k_b[0],startTime) for k_b in enumerate(miniBatches)]
            totalN = self.totalNumExamples(miniBatches)
            logging.info("created %d minibatch tasks, total of %d examples" % (len(bpInputs),totalN))
            #generate gradients - in parallel
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)
            #update params using the gradients
            logging.info("gradients for %d minibatch tasks computed" % len(bpInputs))
            self.processGradients(bpOutputs,totalN)
            logging.info("gradients merged")
            # send params to workers
            self.broadcastParameters()
            logging.info("parameters broadcast to workers")
            # status updates
            epochCounter = learn.GradAccumulator.mergeCounters( [n_grads[1].counter for n_grads in bpOutputs] )
            self.epochTracer(self,epochCounter,i=i,startTime=trainStartTime)

class ParallelAdaGradLearner(ParallelFixedRateGDLearner):
    """ Not debugged yet....
    """
    
    #override learning rate
    def __init__(self,prog,**kw):
        logging.warn("ParallelAdaGradLearner does not seem to perform well - it's probably still buggy")
        if not 'rate' in kw: kw['rate']=0.5
        super(ParallelAdaGradLearner,self).__init__(prog,**kw)

    def train(self,dset):
        modes = dset.modesToLearn()
        trainStartTime = time.time()
        sumSquareGrads = learn.GradAccumulator()
        for i in range(self.epochs):

            logging.info("starting epoch %d" % i)
            startTime = time.time()
            #generate the tasks
            miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize))
            bpInputs = [ParallelFixedRateGDLearner.miniBatchToTask(k_b[1],i,k_b[0],startTime) for k_b in enumerate(miniBatches)]
            totalN = self.totalNumExamples(miniBatches)

            #generate gradients - in parallel
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)

            # accumulate to sumSquareGrads
            totalGradient = learn.GradAccumulator()
            for (n,paramGrads) in bpOutputs:
                for (functor,arity),grad in list(paramGrads.items()):
                    totalGradient.accum((functor,arity), self.meanUpdate(functor,arity,grad,n,totalN))
            sumSquareGrads = sumSquareGrads.addedTo(totalGradient.mapData(NP.square))
            #compute gradient-specific rate
            ratePerParam = sumSquareGrads.mapData(lambda d:d+1e-1).mapData(NP.sqrt).mapData(NP.reciprocal)

            # scale down totalGradient by per-feature weight
            for (functor,arity),grad in list(totalGradient.items()):
                totalGradient[(functor,arity)] = grad.multiply(ratePerParam[(functor,arity)])

            self.regularizer.regularizeParams(self.prog,totalN)
            for (functor,arity) in self.prog.db.paramList:
                m = self.prog.db.getParameter(functor,arity)
                print(('reg',functor,'/',arity,'m shape',m.shape))
                if (functor,arity) in list(totalGradient.keys()):
                    print(('vs totalGradient shape',totalGradient[(functor,arity)].shape))
                else:
                    print('not in totalGradient')

            #cannot use process gradients because I've already scaled them down,
            # need to just add and clip
            for (functor,arity),grad in list(totalGradient.items()):
                m0 = self.prog.db.getParameter(functor,arity)
                m1 = m0 + self.rate * grad
                m = mutil.mapData(lambda d:NP.clip(d,0.0,NP.finfo('float32').max), m1)
                self.prog.db.setParameter(functor,arity,m)

            # send params to workers
            self.broadcastParameters()

            # status updates
            epochCounter = learn.GradAccumulator.mergeCounters( [n_grads[1].counter for n_grads in bpOutputs] )
            self.epochTracer(self,epochCounter,i=i,startTime=trainStartTime)
