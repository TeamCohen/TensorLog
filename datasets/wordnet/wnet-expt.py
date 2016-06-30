import sys

import os.path
import matrixdb
import expt
import dataset
import declare
import tensorlog
import ops
import learn

if __name__=="__main__":
    pred = 'hypernym' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 10 if len(sys.argv)<=2 else int(sys.argv[2])
        
    print '== pred',pred,'epochs',epochs
    db = matrixdb.MatrixDB.uncache('wnet.db','wnet.cfacts')
    trainData = dataset.Dataset.uncacheExamples('%s-train.dset' % pred,db,'%s-train.examples' % pred,proppr=True)
    testData = dataset.Dataset.uncacheExamples('%s-test.dset' % pred,db,'%s-test.examples' % pred,proppr=True)
    prog = tensorlog.ProPPRProgram.load(["wnet-learned.ppr"],db=db)
    prog.setWeights(db.vector(declare.asMode("rule(i)")))
    def learnerFactory(prog):
        return learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),traceFun=learn.Learner.cheapTraceFun,epochs=10)

    #ops.conf.trace = True
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'targetPred':'i_%s/io' % pred,
              'savedTestPreds':'%s-test.solutions.txt' % pred,
              'savedTrainExamples':'%s-train.examples' % pred,
              'savedTestExamples':'%s-test.examples' % pred,
              'learnerFactory':learnerFactory,
    }
    expt.Expt(params).run()
