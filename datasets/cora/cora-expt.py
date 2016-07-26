import logging

import expt
import tensorlog
import learn
import plearn

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info('level is info')

    db = tensorlog.parseDBSpec('tmp-cache/cora.db|inputs/cora.cfacts')
    trainData = tensorlog.parseDatasetSpec('tmp-cache/cora-train.dset|inputs/train.examples', db)
    testData = tensorlog.parseDatasetSpec('tmp-cache/cora-test.dset|inputs/test.examples', db)
    prog = tensorlog.parseProgSpec("cora.ppr",db,proppr=True)
    prog.setRuleWeights()
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 1
#    learner = learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),epochs=5)
    learner = plearn.ParallelFixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),parallel=5,epochs=5)
    params = {'prog':prog,
              'trainData':trainData, 'testData':testData,
              'targetMode':'samebib/io',
              'savedModel':'tmp-cache/cora-trained.db',
              'savedTestPredictions':'tmp-cache/cora-test.solutions.txt',
              'savedTrainExamples':'tmp-cache/cora-train.examples',
              'savedTestExamples':'tmp-cache/cora-test.examples',
              'learner':learner
    }
    print 'maxdepth',prog.maxDepth
    expt.Expt(params).run()
