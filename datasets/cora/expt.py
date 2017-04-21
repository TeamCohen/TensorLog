import logging

from tensorlog import expt
from tensorlog import learn
from tensorlog import plearn
from tensorlog import comline

def setExptParams():
    db = comline.parseDBSpec('tmp-cache/cora.db|inputs/cora.cfacts')
    trainData = comline.parseDatasetSpec('tmp-cache/cora-train.dset|inputs/train.examples', db)
    testData = comline.parseDatasetSpec('tmp-cache/cora-test.dset|inputs/test.examples', db)
    prog = comline.parseProgSpec("cora.ppr",db,proppr=True)
    prog.setRuleWeights()
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 1
    learner = plearn.ParallelFixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),parallel=5,epochs=30)
    return {'prog':prog,
            'trainData':trainData, 'testData':testData,
            'targetMode':'samebib/io',
            'savedModel':'tmp-cache/cora-trained.db',
            'savedTestPredictions':'tmp-cache/cora-test.solutions.txt',
            'savedTrainExamples':'tmp-cache/cora-train.examples',
            'savedTestExamples':'tmp-cache/cora-test.examples',
            'learner':learner
    }

def runMain():
    logging.basicConfig(level=logging.INFO)
    params = setExptParams()
    return expt.Expt(params).run()

if __name__=="__main__":
  acc,loss = runMain()
  print 'acc,loss',acc,loss
