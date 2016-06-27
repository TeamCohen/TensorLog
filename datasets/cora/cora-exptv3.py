nimport os.path
import scipy.sparse as SS
import scipy.io

# trying to get cora to work on recursive theories
#  - main problem: old theory leads to huge proof counts,
#    fixed by refactoring theory to be linear.
#  - also messed with gradient clipping and regularizer

import declare
import bpcompiler
import exptv2
import dataset
import tensorlog
import matrixdb
import funs
import ops
import mutil
import learn

if __name__=="__main__":
    db = matrixdb.MatrixDB.uncache('tmp-cache/cora-linear.db','inputs/cora-linear.cfacts')
    trainData = dataset.Dataset.uncacheExamples('tmp-cache/cora-linear-train.dset',db,'inputs/train.examples')
    testData = dataset.Dataset.uncacheExamples('tmp-cache/cora-linear-test.dset',db,'inputs/test.examples')
    print 'train:','\n  '.join(trainData.pprint())
    print 'test: ','\n  '.join(testData.pprint())
    prog = tensorlog.ProPPRProgram.load(["cora-linear.ppr"],db=db)
    initWeights = prog.db.vector(declare.asMode("rule(o)"))
    prog.setWeights(initWeights)
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 3
    ops.conf.optimize_component_multiply = True
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'targetPred':'samebib/io',
              'savedModel':'tmp-cache/cora-linear-trained.db',
              'savedTestPreds':'tmp-cache/cora-test.solutions.txt',
              'savedTrainExamples':'tmp-cache/cora-linear-train.examples',
              'savedTestExamples':'tmp-cache/cora-linear-test.examples',
              #'regularizer':learn.L2Regularizer(0.1),
    }
    print 'maxdepth',prog.maxDepth
    exptv2.Expt(params).run()
