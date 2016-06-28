import sys

import os.path
import matrixdb
import expt
import dataset
import declare
import tensorlog
import ops
        
if __name__=="__main__":
    if len(sys.argv)<=1:
        pred = 'hypernym'
    else:
        pred = sys.argv[1]
    print '== pred',pred
    db = matrixdb.MatrixDB.uncache('wnet.db','wnet.cfacts')
    trainData = dataset.Dataset.uncacheExamples('%s-train.dset' % pred,db,'%s-train.examples' % pred,proppr=True)
    testData = dataset.Dataset.uncacheExamples('%s-test.dset' % pred,db,'%s-test.examples' % pred,proppr=True)
    prog = tensorlog.ProPPRProgram.load(["wnet-learned.ppr"],db=db)
    prog.setWeights(db.vector(declare.asMode("rule(i)")))

    #ops.conf.trace = True
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'targetPred':'i_%s/io' % pred,
              'savedTestPreds':'%s-test.solutions.txt' % pred,
              'savedTrainExamples':'%s-train.examples' % pred,
              'savedTestExamples':'%s-test.examples' % pred,
              'epochs':30,
    }
    expt.Expt(params).run()
