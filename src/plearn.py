# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# parallel learning implementation(s)
#

import time
import mutil
import learn
import multiprocessing.pool    

#TODO should trace function report the worker pid?

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
    paramGrads = workerLearner.crossEntropyGrad(mode,X,Y,traceFunArgs=args)
    return (mutil.numRows(X),paramGrads)

##############################################################################
# A parallel learner.
##############################################################################

class ParallelFixedRateGDLearner(learn.FixedRateSGDLearner):
    """Split task into fixed-size miniBatchs and compute gradients of
    these in parallel.  Parameter updates are done only at the end of
    an epoch so this is actually gradient descent, not SGD.
    """

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None,miniBatchSize=100,parallel=10):
        super(ParallelFixedRateGDLearner,self).__init__(
            prog,epochs=epochs,rate=rate,regularizer=regularizer,miniBatchSize=miniBatchSize)
        self.traceFun = traceFun or learn.FixedRateSGDLearner.defaultTraceFun
        print '= pool initialized with',parallel,'processes'
        #initargs are used to build a worker learner for the pool,
        #which just computes the gradients and does nothing else
        self.pool = multiprocessing.pool.Pool(
            parallel, initializer=_initWorker, 
            initargs=(learn.FixedRateSGDLearner,self.prog,self.epochs,self.rate,self.regularizer,self.traceFun,self.miniBatchSize))
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        for i in range(self.epochs):
            miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize))
            totalN = sum(mutil.numRows(X) for (mode,X,Y) in miniBatches)
            def miniBatchToTask(k):
                (mode,X,Y) = miniBatches[k]
                args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
                return (mode,X,Y,args)
            #generate the tasks
            bpInputs = map(miniBatchToTask, range(len(miniBatches)))
            print '= built tasks:',time.time()-startTime,'cum sec'
            #generate gradients - in parallel
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)
            print '= ran parallel bp :',time.time()-startTime,'cum sec'
            #apply the gradient
            for (n,paramGrads) in bpOutputs:
                # self.regularizer.addRegularizationGrad(paramGrads,self.prog)
                # scale rate down to reflect the fraction of the data  
                self.applyMeanUpdate(paramGrads, (self.rate*n)/totalN, totalN)

            print '= applied gradient :',time.time()-startTime,'cum sec'


