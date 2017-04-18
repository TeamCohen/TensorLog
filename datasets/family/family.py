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

#logging.basicConfig(level=logging.DEBUG)

CROSS_COMPILE = []
CROSS_LEARN = {}

try:
    from tensorlog import theanoxcomp
    CROSS_COMPILE.append(theanoxcomp.DenseMatDenseMsgCrossCompiler)
    CROSS_LEARN[theanoxcomp.DenseMatDenseMsgCrossCompiler] = theanoxcomp.FixedRateGDLearner
    CROSS_COMPILE.append(theanoxcomp.SparseMatDenseMsgCrossCompiler)
    CROSS_LEARN[theanoxcomp.SparseMatDenseMsgCrossCompiler] = theanoxcomp.FixedRateGDLearner
except:
    pass

stem = "kinship"
if __name__=="__main__":
    if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
    db = comline.parseDBSpec('tmp-cache/{stem}.db|inputs/{stem}.cfacts:inputs/{stem}-rule.cfacts'.format(stem=stem))
    trainData = comline.parseDatasetSpec('tmp-cache/{stem}-train.dset|inputs/{stem}-train.examples'.format(stem=stem),db)
    testData = comline.parseDatasetSpec('tmp-cache/{stem}-test.dset|inputs/{stem}-test.examples'.format(stem=stem),db)
    print 'train:','\n  '.join(trainData.pprint())
    print 'test: ','\n  '.join(testData.pprint())
    #dTrain = uncacheMatPairs('%s.db' % stem,'raw/%s.train.examples' % stem)
    #dTest = uncacheMatPairs('%s.db' % stem,'raw/%s.test.examples' % stem)
    prog = program.ProPPRProgram.loadRules("%s-train-isg.ppr" % stem,db=db)
    prog.setRuleWeights()
    prog.maxDepth=4
    params = {'prog':prog,
              #'theoryPred':'concept_atdate',
              'trainData':trainData,
              'testData':testData,
              'savedModel':'tmp-cache/%s-trained.db' % stem,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % stem,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
    }
    expt.Expt(params).run()
    
    for compilerClass in CROSS_COMPILE:
        
        print compilerClass
        xc = compilerClass(prog)
        # compile everything
        for mode in trainData.modesToLearn():
          xc.ensureCompiled(mode)
        learner = CROSS_LEARN[compilerClass](prog,xc)
        
        params = {'prog':prog,
                  'trainData':trainData, 'testData':testData,
                  'savedTestPredictions':'tmp-cache/%s-test.%s.solutions.txt' % (stem,compilerClass.__name__),
                  'savedTestExamples':'tmp-cache/%s-test.%s.examples' % (stem,compilerClass.__name__),
                  'learner':learner,
        }
        
        testAcc,testXent = expt.Expt(params).run()
