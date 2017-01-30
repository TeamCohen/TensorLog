import os
import scipy.sparse as SS
import scipy.io

from tensorlog import expt
from tensorlog import dataset
from tensorlog import comline
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import logging
from tensorlog import funs

#logging.basicConfig(level=logging.DEBUG)

stem = "kinship"
if __name__=="__main__":
    if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
    db = matrixdb.MatrixDB.uncache('tmp-cache/%s.db' % stem,
        'inputs/%s.cfacts' % stem, 
        'inputs/%s-rule.cfacts' % stem)
    trainData = dataset.Dataset.uncacheExamples('tmp-cache/%s-train.dset' % stem,db,'inputs/%s-train.examples'%stem)
    testData = dataset.Dataset.uncacheExamples('tmp-cache/%s-test.dset'%stem,db,'inputs/%s-test.examples'%stem)
    print 'train:','\n  '.join(trainData.pprint())
    print 'test: ','\n  '.join(testData.pprint())
    #dTrain = uncacheMatPairs('%s.db' % stem,'raw/%s.train.examples' % stem)
    #dTest = uncacheMatPairs('%s.db' % stem,'raw/%s.test.examples' % stem)
    prog = program.ProPPRProgram.load(["%s-train-isg.ppr" % stem],db=db)
    prog.setRuleWeights()
    prog.maxDepth=4
    params = {'initProgram':prog,
              #'theoryPred':'concept_atdate',
              'trainData':trainData,
              'testData':testData,
              'savedModel':'tmp-cache/%s-trained.db' % stem,
              'savedTestPreds':'tmp-cache/%s-test.solutions.txt' % stem,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
    }
    expt.Expt(params).run()
