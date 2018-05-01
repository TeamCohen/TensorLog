import logging
import getopt
import sys

from tensorlog import masterconfig
from tensorlog import expt
from tensorlog import learn
from tensorlog import comline
from tensorlog import matrixdb
from tensorlog.helper.sketchadapters import sketchProgram,SketchData,SketchLearner
from tensorlog.helper.fast_sketch import FastSketcher,FastSketcher2
from tensorlog.helper.countmin_embeddings import Sketcher,Sketcher2
SKETCHERS={
    "FastSketcher2":FastSketcher2,
    "FastSketcher":FastSketcher,
    "Sketcher":Sketcher,
    "Sketcher2":Sketcher2
    }

import os
import psutil
process = psutil.Process(os.getpid())
def printMem(msg):
  print "MEMORY",msg
  print "MEMORY",process.memory_info()[0]

import expt as dflt


def setExptParams(num):
    matrixdb.conf.ignore_types = True
    db = comline.parseDBSpec('tmp-cache/sketch-train-%d.db|inputs/train-%d.cfacts' % (num,num))
    trainData_native = comline.parseDatasetSpec('tmp-cache/sketch-train-%d.dset|inputs/train-%d.exam'  % (num,num), db)
    testData_native = comline.parseDatasetSpec('tmp-cache/sketch-test-%d.dset|inputs/test-%d.exam'  % (num,num), db)

    optlist,args = getopt.getopt(sys.argv[1:],'x',"k= delta= sketcher=".split())
    optdict = dict(optlist)
    print 'optdict',optdict
    k=optdict.get('--k','160')
    delta = optdict.get('--delta','0.01')
    sketcher_s = optdict.get('--sketcher','FastSketcher2')
    if sketcher_s not in SKETCHERS:
        print "no sketcher named %s" % sketcher_s
        print "available sketchers:"
        print "\n".join(SKETCHERS.keys())
        exit(0)
    sketcher_c = SKETCHERS[sketcher_s]
    
    sketcher = sketcher_c(db,int(k),float(delta)/db.dim(),verbose=False)
    sketcher.describe()

    trainData = SketchData(sketcher,trainData_native)
    testData = SketchData(sketcher,testData_native)
    
    prog = comline.parseProgSpec("theory.ppr",db,proppr=True)
    sketchProgram(sketcher,prog)
    prog.setFeatureWeights()
    learner = SketchLearner(sketcher,[trainData,testData],prog,regularizer=learn.L2Regularizer(),epochs=10,epochTracer=memoryTrace)
    return {'prog':prog,
            'trainData':trainData,
            'testData':testData,
            #'targetMode':'answer/io',
            'savedModel':'learned-model.db',
            'learner':learner
    }

def memoryTrace(learner,ctr,i=-1,**kw):
    printMem("epoch %d tot.time %g"%(i,ctr[('time','tot')]))

def runMain(num=250):
    printMem("baseline")
    params = setExptParams(num)
    printMem("params loaded")
    result =  expt.Expt(params).run()
    printMem("experiment complete")
    return result

def runNative(num=250):
    matrixdb.conf.ignore_types = False
    printMem("baseline")
    params = dflt.setExptParams(num)
    printMem("params loaded")
    params['learner'] = learn.FixedRateGDLearner(params['prog'],regularizer=learn.L2Regularizer(),epochs=10,epochTracer=memoryTrace)
    result = expt.Expt(params).run()
    printMem("experiment complete")
    return result
        
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    masterconfig.masterConfig().matrixdb.allow_weighted_tuples=False
    acc,loss = runMain() # expect 0.21,0.22
    print 'acc,loss',acc,loss
    acc,loss = runNative()
    print 'acc,loss',acc,loss
