import os
import scipy.sparse as SS
import scipy.io

from tensorlog import expt
from tensorlog import dataset
from tensorlog import comline
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import program
from tensorlog import funs
from tensorlog import learn
from tensorlog import declare

stem = "kinship"
def setExptParams():
    db = comline.parseDBSpec('tmp-cache/{stem}.db|inputs/{stem}.cfacts:inputs/{stem}-rule.cfacts'.format(stem=stem))
    trainData = comline.parseDatasetSpec('tmp-cache/{stem}-train.dset|inputs/{stem}-train.examples'.format(stem=stem),db)
    testData = comline.parseDatasetSpec('tmp-cache/{stem}-test.dset|inputs/{stem}-test.examples'.format(stem=stem),db)
    #print 'train:','\n  '.join(trainData.pprint())
    #print 'test: ','\n  '.join(testData.pprint())
    prog = program.ProPPRProgram.loadRules("theory.ppr",db=db)
    prog.setRuleWeights()
    prog.maxDepth=4
    return (prog, trainData, testData)

def runMain():
    if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
    (prog, trainData, testData) = setExptParams()
    mode = declare.asMode("i_niece/io")
    trainData = trainData.extractMode(mode)
    testData = testData.extractMode(mode)
    learner = ProbeLearner(prog)
    probe(learner,prog,testData,mode)

def probe(learner,prog,data,mode):
    #from tensorlog import ops
    #ops.conf.trace=True
    #ops.conf.long_trace=6
    from tensorlog import funs
    #funs.conf.trace=True
    #funs.conf.long_trace=True
    U=learner.datasetPredict(data)
    printSolutions(U,prog,mode)
    
def printSolutions(U,prog,mode):
    dx = prog.db.matrixAsSymbolDict(U.getX(mode))
    dp = prog.db.matrixAsSymbolDict(U.getY(mode))
    n=max(dx.keys())
    for i in range(n+1):
        x = dx[i].keys()[0]
        print "%s(%s,X1)" % (mode.getFunctor(),x)
        scored = reversed(sorted([(py,y) for (y,py) in dp[i].items()]))
        for (r, (py,y)) in enumerate(scored):
            print "%d\t%.18f\t%s(%s,%s)." % (r+1,py,mode.getFunctor(),x,y)
    
def accExpt(prog,trainData,testData):
    params = {'prog':prog,
              'trainData':trainData,
              'testData':testData,
              'savedModel':'tmp-cache/%s-trained.db' % stem,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % stem,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
    }
    return expt.Expt(params).run()

class ProbeLearner(learn.FixedRateGDLearner):
    def __init__(self, prog, **kwargs):
        super(ProbeLearner,self).__init__(prog, **kwargs)
    def predict(self,mode,X,pad=None):
        P = super(ProbeLearner,self).predict(mode,X,pad)
        print mode,mutil.pprintSummary(P)
        return P

if __name__=="__main__":
    runMain()
