# code for running scalability experiments in JAIR submission

import sys
import numpy as NP
import random
import math
import time
import scipy

from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import expt
from tensorlog import funs
from tensorlog import interp
from tensorlog import learn
from tensorlog import matrixdb
from tensorlog import ops
from tensorlog import plearn
from tensorlog import program
from tensorlog import simple

EDGE_WEIGHT = 0.2
SUBGRID = 10

def nodeName(i,j):
    return '%d,%d' % (i,j)

def generateGrid(n,outf):
    fp = open(outf,'w')
    for i in range(1,n+1):
        for j in range(1,n+1):
            for di in [-1,0,+1]:
                for dj in [-1,0,+1]:
                    if (1 <= i+di <= n) and (1 <= j+dj <= n):
                        fp.write('edge\t%s\t%s\t%f\n' % (nodeName(i,j),nodeName(i+di,j+dj),EDGE_WEIGHT))

def generateData(n,trainFile,testFile):
    fpTrain = open(trainFile,'w')
    fpTest = open(testFile,'w')
    r = random.Random()
    for i in range(1,n+1):
        for j in range(1,n+1):
            #target - note early version used i,j < n/2 which is a bug
            ti = (i/SUBGRID)*SUBGRID + SUBGRID/2
            tj = (j/SUBGRID)*SUBGRID + SUBGRID/2
            x = nodeName(i,j)
            y = nodeName(ti,tj)
            fp = fpTrain if r.random()<0.67 else fpTest
            fp.write('\t'.join(['path',x,y]) + '\n')

# parse command line args
def getargs():
    goal = 'acc'
    if len(sys.argv)>1:
        goal = sys.argv[1]
    n = 6
    if len(sys.argv)>2:
        n = int(sys.argv[2])
    maxD = round(n/2.0)
    if len(sys.argv)>3:
        maxD = int(sys.argv[3])
    epochs = 30
    if len(sys.argv)>4:
        epochs = int(sys.argv[4])
    return (goal,n,maxD,epochs)

# generate all inputs for an accuracy (or timing) experiment
def genInputs(n):
    #generate grid
    stem = 'inputs/g%d' % n

    factFile = stem+'.cfacts'
    trainFile = stem+'-train.exam'
    testFile = stem+'-test.exam'

    generateGrid(n,factFile)
    generateData(n,trainFile,testFile)
    return (factFile,trainFile,testFile)

# run timing experiment
def timingExpt(prog,maxD,trainFile,minibatch):
    times = []
    print 'depth',maxD,'minibatch',minibatch
    ti = interp.Interp(prog)
    ti.prog.maxDepth = maxD
    tlog = simple.Compiler(db=prog.db,prog=prog)
    dset = tlog.load_dataset(trainFile)
    if minibatch:
        batchSize = minibatch
        quitAfter = 1
    else:
        batchSize = 1
        quitAfter = 25
    start = time.time()
    for k,(mode,(X0,Y0)) in enumerate(tlog.minibatches(dset,batch_size=batchSize)):
        print 'batch',k
        X = scipy.sparse.csr_matrix(X0)
        Y = scipy.sparse.csr_matrix(Y0)
        ti.prog.eval(declare.asMode(mode), [X])
        if k>=quitAfter:
            break
    elapsed = time.time() - start
    print k*batchSize,'examples','miniBatchSize',batchSize,'time',elapsed,'qps',k*batchSize/elapsed
    return elapsed

# run accuracy experiment
def accExpt(prog,trainFile,testFile,n,maxD,epochs):
    print 'grid-acc-expt: %d x %d grid, %d epochs, maxPath %d' % (n,n,epochs,maxD)
    trainData = dataset.Dataset.loadExamples(prog.db,trainFile)
    testData = dataset.Dataset.loadExamples(prog.db,testFile)
    prog.db.markAsParameter('edge',2)
    prog.maxDepth = maxD
    # 20 epochs and rate=0.01 is ok for grid size 16 depth 10
    # then it gets sort of chancy
    #learner = learn.FixedRateGDLearner(prog,epochs=epochs,epochTracer=learn.EpochTracer.cheap)
    learner = learn.FixedRateGDLearner(prog,epochs=epochs,epochTracer=learn.EpochTracer.cheap,rate=0.005)
    plearner = plearn.ParallelFixedRateGDLearner(
        prog,
        epochs=epochs,
        parallel=40,
        miniBatchSize=BATCHSIZE,
        regularizer=learn.L2Regularizer(),
        epochTracer=learn.EpochTracer.cheap,
        rate=0.01)
    params = {'prog':prog,
              'trainData':trainData, 'testData':testData,
              'savedTestPredictions':'tmp-cache/test.solutions.txt',
              'savedTestExamples':'tmp-cache/test.examples',
              'learner':learner,
    }
    NP.seterr(divide='raise')
    t0 = time.time()
    result =  expt.Expt(params).run()
    print 'elapsed time',time.time()-t0
    return result

def runMain():

    # usage: acc [grid-size] [maxDepth] [epochs]"
    #        time [grid-size] [maxDepth] [no-minibatch]"
    (goal,n,maxD,epochsOrMinibatch) = getargs()
    print 'args',(goal,n,maxD,epochsOrMinibatch)
    (factFile,trainFile,testFile) = genInputs(n)

    db = matrixdb.MatrixDB.loadFile(factFile)
    prog = program.Program.loadRules("grid.ppr",db)

    if goal=='time':
        print timingExpt(prog,maxD,trainFile,epochsOrMinibatch)
    elif goal=='acc':
        print accExpt(prog,trainFile,testFile,n,maxD,epochsOrMinibatch)
        print 'prog.maxDepth',prog.maxDepth
    else:
        assert False,'bad goal %s' % goal

if __name__=="__main__":
    runMain()
