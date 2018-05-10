import sys
import numpy as NP
import random
import math
import time

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

random.seed(31415926)
#funs.conf.trace=True
#ops.conf.trace=True

from expt import *

def generateGrid(n,blockSize,outf):
    fp = open(outf,'w')
    for i in range(1,blockSize+1):
        for j in range(1,(n*blockSize)+1):
            for di in [-1,0,+1]:
                for dj in [-1,0,+1]:
                    if (1 <= i+di <= blockSize) and (1 <= j+dj <= (n*blockSize)):
                        fp.write('edge\t%s\t%s\t%f\n' % (nodeName(i,j),nodeName(i+di,j+dj),EDGE_WEIGHT))

def generateData(n,blockSize,trainFile,testFile):
    fpTrain = open(trainFile,'w')
    fpTest = open(testFile,'w')
    r = random.Random()
    for i in range(1,blockSize+1):
        for j in range(1,(n*blockSize)+1):
            #target:
            # e.g. if blockSize is 10, 
            # assign 1-5 to 1, 6-10 to 10,
            # 11-15 to 11, 16-20 to 20, etc
            ti = 1 if i<=blockSize/2 else blockSize
            d = int((j-1)/blockSize)
            tj = d*blockSize+1 if (j-1)%blockSize<blockSize/2 else (d+1)*blockSize
            
            x = nodeName(i,j)
            y = nodeName(ti,tj)
            fp = fpTrain if r.random()<0.67 else fpTest
            fp.write('\t'.join(['path',x,y]) + '\n')

# generate all inputs for an accuracy (or timing) experiment
def genInputs(n,blockSize):
    #generate grid
    stem = 'inputs/g%dx%d' % (blockSize,n)

    factFile = stem+'.cfacts'
    trainFile = stem+'-train.exam'
    testFile = stem+'-test.exam'

    generateGrid(n,blockSize,factFile)
    generateData(n,blockSize,trainFile,testFile)
    return (factFile,trainFile,testFile)

# run accuracy experiment
def accExpt(prog,trainFile,testFile,n,maxD,epochs):
    print 'grid-acc-expt: %d x %d grid, %d epochs, maxPath %d' % (maxD,n*maxD,epochs,maxD)
    trainData = dataset.Dataset.loadExamples(prog.db,trainFile)
    testData = dataset.Dataset.loadExamples(prog.db,testFile)
    prog.db.markAsParameter('edge',2)
    prog.maxDepth = maxD
    # 20 epochs and rate=0.01 is ok for grid size 16 depth 10
    # then it gets sort of chancy
    #learner = learn.FixedRateGDLearner(prog,epochs=epochs,epochTracer=learn.EpochTracer.cheap)
    learner = learn.FixedRateGDLearner(prog,epochs=epochs,epochTracer=learn.EpochTracer.cheap,rate=0.01)
    params = {'prog':prog,
              'trainData':trainData, 'testData':testData,
              'savedTestPredictions':'tmp-cache/test.solutions.txt',
              'savedTestExamples':'tmp-cache/test.examples',
              'learner':learner,
    }
    NP.seterr(divide='raise')
    result =  expt.Expt(params).run()
    return result

def runMain(accExpt_fun=accExpt):
    if len(sys.argv)<3:
      print "usage:"
      print "  acc [grid-size] [max-depth] [epochs]"
      print "build [grid-size] [max-depth]"
      exit(0)
    (goal,n,maxD,epochs) = getargs()
    print 'args',(goal,n,maxD,epochs)
    (factFile,trainFile,testFile) = genInputs(n,maxD)

    db = matrixdb.MatrixDB.loadFile(factFile)
    prog = program.Program.loadRules("grid.ppr",db)

    if goal=='acc':
        print "acc",accExpt_fun(prog,trainFile,testFile,n,maxD,epochs)
    elif goal=='build':
      pass
    else:
        assert False,'bad goal %s' % goal

if __name__=="__main__":
    runMain()
