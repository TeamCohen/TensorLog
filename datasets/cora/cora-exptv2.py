import os.path
import scipy.sparse as SS
import scipy.io

# based on new experiment package, mostly in dataset.py - does not
# give same results as old cora experiment.


import exptv2
import dataset
import tensorlog
import matrixdb
import funs
import ops
import mutil

if __name__=="__main__":
    db = matrixdb.MatrixDB.uncache('tmp-cache/cora.db','inputs/cora.cfacts')
    trainData = dataset.Dataset.uncacheExamples('tmp-cache/cora-train.dset',db,'inputs/train.examples')
    testData = dataset.Dataset.uncacheExamples('tmp-cache/cora-test.dset',db,'inputs/test.examples')
    print 'train:','\n  '.join(trainData.pprint())
    print 'test: ','\n  '.join(testData.pprint())
    prog = tensorlog.ProPPRProgram.load(["cora.ppr"],db=db)
    prog.setWeights(db.ones())
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 1
    ops.conf.optimize_component_multiply = True
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'targetPred':'samebib/io',
              'savedModel':'tmp-cache/cora-trained.db',
              'savedTestPreds':'tmp-cache/cora-test.solutions.txt',
              'savedTrainExamples':'tmp-cache/cora-train.examples',
              'savedTestExamples':'tmp-cache/cora-test.examples',
    }
    print 'maxdepth',prog.maxDepth
    exptv2.Expt(params).run()
