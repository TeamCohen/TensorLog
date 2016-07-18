# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# parallel learning implementation(s)
#

import os
import time
import multiprocessing
import multiprocessing.pool    
import logging

import mutil
import learn

#TODO iterate don't make lists

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
    for (functor,arity),value in paramDict.items():
        workerLearner.prog.db.setParameter(functor,arity,value)

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

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,tracer=None,miniBatchSize=100,parallel=10,epochTracer=None):
        super(ParallelFixedRateGDLearner,self).__init__(
            prog,epochs=epochs,rate=rate,regularizer=regularizer,miniBatchSize=miniBatchSize,tracer=tracer)
        self.epochTracer = epochTracer or learn.EpochTracer.default
        self.parallel = parallel or multiprocessing.cpu_count()
        logging.info('pool initialized with %d processes' % self.parallel)
        #initargs are used to build a worker learner for the pool,
        #which just computes the gradients and does nothing else
        self.pool = multiprocessing.pool.Pool(
            self.parallel, 
            initializer=_initWorker, 
            #crucial to get the argument order right here!
            initargs=(learn.FixedRateSGDLearner,self.prog,self.epochs,self.rate,self.regularizer,self.tracer,self.miniBatchSize))
        logging.info('created pool of %d workers' % parallel)
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        for i in range(self.epochs):
            logging.info('preparing minibatches')
            miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize))
            totalN = sum(mutil.numRows(X) for (mode,X,Y) in miniBatches)
            def miniBatchToTask(k):
                (mode,X,Y) = miniBatches[k]
                args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
                return (mode,X,Y,args)

            #generate the tasks
            bpInputs = map(miniBatchToTask, range(len(miniBatches)))
            logging.info('created %d minibatch tasks' % len(bpInputs))

            #generate gradients - in parallel
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)
            logging.info('produced %d minibatch gradients' % len(bpOutputs))

            #apply the gradient
            for (n,paramGrads) in bpOutputs:
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                # scale rate down to reflect the fraction of the data  
                self.applyMeanUpdate(paramGrads, self.rate, n, totalN)
            logging.info('performed param updates on master')

            #broadcast the new parameters to the subprocesses 
            paramDict = dict(
                ((functor,arity),self.prog.db.getParameter(functor,arity))
                for (functor,arity) in self.prog.db.params)
            #send one _doAcceptNewParams task to each worker. warning:
            # this is not guaranteed to work from the API 
            self.pool.map(_doAcceptNewParams, [paramDict]*self.parallel, chunksize=1)
            logging.info('broadcast param updates to workers')

            self.epochTracer(self,dset,i=i,startTime=startTime)


