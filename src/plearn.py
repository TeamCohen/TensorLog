import time
import mutil
import learn
import multiprocessing.dummy

def _doBackpropTask(task):
    (learner,mode,X,Y,args) = task
    n = mutil.numRows(X)
    paramGrads = learner.crossEntropyGrad(mode,X,Y,traceFunArgs=args)
    return (n,paramGrads)

class ModeParallelFixedRateGDLearner(learn.FixedRateGDLearner):

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None,parallel=10):
        super(ModeParallelFixedRateGDLearner,self).__init__(prog,regularizer=regularizer,traceFun=traceFun,epochs=epochs,rate=rate)
        self.pool = multiprocessing.dummy.Pool(parallel)
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        for i in range(self.epochs):
            args = {'i':i,'startTime':startTime}
            def buildTask(mode):
                modeArgs = dict(args.items())
                modeArgs['mode']=mode
                return (self,mode,dset.getX(mode),dset.getY(mode),modeArgs)
            bpInputs = map(buildTask, dset.modesToLearn())
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)
            for (n,paramGrads) in bpOutputs:
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                self.applyMeanUpdate(paramGrads,self.rate,n)


class MinibatchParallelFixedRateGDLearner(learn.FixedRateSGDLearner):

    def __init__(self,prog,epochs=10,rate=0.1,regularizer=None,traceFun=None,miniBatchSize=100,parallel=10):
        super(MinibatchParallelFixedRateGDLearner,self).__init__(
            prog,epochs=epochs,rate=rate,regularizer=regularizer,miniBatchSize=miniBatchSize)
        self.traceFun = traceFun or learn.FixedRateSGDLearner.defaultTraceFun
        self.pool = multiprocessing.dummy.Pool(parallel)
    
    def train(self,dset):
        startTime = time.time()
        modes = dset.modesToLearn()
        for i in range(self.epochs):
            miniBatches = list(dset.minibatchIterator(batchSize=self.miniBatchSize))
            def miniBatchToTask(k):
                (mode,X,Y) = miniBatches[k]
                args = {'i':i,'k':k,'startTime':startTime,'mode':mode}
                return (self,mode,X,Y,args)
            bpInputs = map(miniBatchToTask, range(len(miniBatches)))
            bpOutputs = self.pool.map(_doBackpropTask, bpInputs)
            for (n,paramGrads) in bpOutputs:
                self.regularizer.addRegularizationGrad(paramGrads,self.prog,n)
                self.applyMeanUpdate(paramGrads,self.rate,n)
                
