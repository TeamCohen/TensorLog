import os
import sys
import getopt
import scipy.sparse as SS
import scipy.io
import numpy as np
from tensorlog import expt,dataset,comline,matrixdb,mutil,program,funs,sketchcompiler,declare
from tensorlog.helper.sketchadapters import SketchData,SketchLearner,sketchProgram
from tensorlog.helper.fast_sketch import FastSketcher2# as DefaultSketcher
from tensorlog.helper.fast_sketch import FastSketcher# as DefaultSketcher
from tensorlog.helper.countmin_embeddings import Sketcher# as DefaultSketcher
from tensorlog.helper.countmin_embeddings import Sketcher2# as DefaultSketcher
import probe as p
import random
random.seed(3141592)

SKETCHERS={
    "FastSketcher2":FastSketcher2,
    "FastSketcher":FastSketcher,
    "Sketcher":Sketcher,
    "Sketcher2":Sketcher2
    }

stem = "kinship"
def setExptParams():
    db = comline.parseDBSpec('tmp-cache/{stem}.db|inputs/{stem}.cfacts:inputs/{stem}-rule.cfacts'.format(stem=stem))
    trainData = comline.parseDatasetSpec('tmp-cache/{stem}-train.dset|inputs/{stem}-train.examples'.format(stem=stem),db)
    testData = comline.parseDatasetSpec('tmp-cache/{stem}-test.dset|inputs/{stem}-test.examples'.format(stem=stem),db)
    #print 'train:','\n  '.join(trainData.pprint())
    #print 'test: ','\n  '.join(testData.pprint())

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
    
    # 'weighted' has outdegree 151 and we're getting too few
    # values out of unsketch if k is too low (?)
    sketcher = sketcher_c(db,int(k),float(delta)/db.dim(),verbose=False)
    sketcher.describe()

    if probe:
        trainData = trainData.extractMode(mode)
        testData = testData.extractMode(mode)
    
    skTrain = SketchData(sketcher,trainData,"train")
    skTest = SketchData(sketcher,testData,"test")
    
    prog = program.ProPPRProgram.loadRules("theory.ppr",db=db)
    sketchProgram(sketcher,prog)
    prog.setRuleWeights()
    prog.maxDepth=4
    return (prog, skTrain,skTest, sketcher, (k,delta,sketcher_s))

mode = declare.asMode("i_brother/io")
probe = False

def runMain():
    if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
    (prog, trainData, testData, sk, args) = setExptParams()

    if not probe:
        print accExpt(prog,trainData,testData,sk,args)
        return (None, prog, testData, mode, sk)
    else:
        insert_sk_funs(prog.getFunction(mode),sk)
        learner = SketchLearner(sk,[trainData,testData],prog,epochs=4)
        p.probe(learner,prog,trainData,mode)
        return (learner,prog,testData,mode,sk)

# this is only needed for generating sensical trace output
def insert_sk_funs(fun,sk):
    if hasattr(fun,'sk'):return
    fun.sk = sk
    if hasattr(fun,'fun'): insert_sk_funs(fun.fun,sk)
    if hasattr(fun,'funs'):
        for f in fun.funs: insert_sk_funs(f,sk)
    
def accExpt(prog,trainData,testData,sketcher,args):
    learner = SketchLearner(sketcher,[trainData,testData],prog,epochs=10)
    params = {'prog':prog,
              'trainData':trainData,
              'testData':testData,
              'savedModel':'tmp-cache/%s-trained.db' % stem,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.%s.txt' % (stem,"-".join(args)),
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
              'learner':learner,
    }
    return expt.Expt(params).run()

if __name__=="__main__":
    data = runMain()
