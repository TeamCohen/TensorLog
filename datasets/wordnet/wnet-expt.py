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
    #usage: [targetPredicate] [epochs]
    pred = 'hypernym' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 30 if len(sys.argv)<=2 else int(sys.argv[2])
    optdict,args = tensorlog.parseCommandLine([
            '--db', 'wnet.db|wnet.cfacts',
            '--prog','wnet-learned.ppr', '--proppr',
            '--train','%s-train.dset|%s-train.examples' % (pred,pred),
            '--test', '%s-test.dset|%s-test.examples' % (pred,pred)])
    prog = optdict['prog']
    prog.setWeights(prog.db.vector(declare.asMode("rule(i)")))
    learner = learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),traceFun=learn.Learner.cheapTraceFun,epochs=epochs)

    #ops.conf.trace = True
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'targetMode':'i_%s/io' % pred,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % pred,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % pred,
              'savedTestExamples':'tmp-cache/%s-test.examples' % pred,
              'learner':learner
    }
    expt.Expt(params).run()
