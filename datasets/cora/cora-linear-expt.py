import logging

import tensorlog
import expt
import learn
import plearn

# a version of cora with left-linear recursion
#
# status: July 13, this runs in about 2min on tanka w/ 22 processes,
# but does a horrible job, 

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info('level is info')

    db = tensorlog.parseDBSpec('tmp-cache/cora-linear.db|inputs/cora-linear.cfacts')
    trainData = tensorlog.parseDatasetSpec('tmp-cache/cora-linear-train.dset|inputs/train.examples', db)
    testData = tensorlog.parseDatasetSpec('tmp-cache/cora-linear-test.dset|inputs/test.examples', db)
    prog = tensorlog.parseProgSpec("cora-linear.ppr",db,proppr=True)
    prog.setRuleWeights()
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    # there will be 22 minibatches by default
    learner = plearn.ParallelFixedRateGDLearner(
        prog,
        regularizer=learn.L2Regularizer(),
        epochs=5,
        miniBatchSize=50,
        parallel=22)
    prog.maxDepth = 3
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
