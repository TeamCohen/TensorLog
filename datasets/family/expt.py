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
    print accExpt(prog,trainData,testData)
    
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

if __name__=="__main__":
    runMain()
