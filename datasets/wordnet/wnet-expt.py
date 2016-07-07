import sys

import expt
import declare
import tensorlog
import learn

if __name__=="__main__":
    #usage: [targetPredicate] [epochs]
    
    #get the command-line options for this experiment
    pred = 'hypernym' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 30 if len(sys.argv)<=2 else int(sys.argv[2])

    # use tensorlog.parseCommandLine to set up the program, etc
    optdict,args = tensorlog.parseCommandLine([
            '--db', 'wnet.db|wnet.cfacts',
            '--prog','wnet-learned.ppr', '--proppr',
            '--train','%s-train.dset|%s-train.examples' % (pred,pred),
            '--test', '%s-test.dset|%s-test.examples' % (pred,pred)])

    # prog is shortcut to the output optdict, for convenience.
    prog = optdict['prog']

    # the weight vector is sparse - just the constants in the unary predicate rule
    prog.setWeights(prog.db.vector(declare.asMode("rule(i)")))

    # use a non-default learner, overriding the tracing function,
    # number of epochs, and regularizer
#    learner = learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),traceFun=learn.Learner.cheapTraceFun,epochs=epochs)
    learner = learn.FixedRateGDLearner(prog,epochs=epochs)

    # configute the experiment
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'targetMode':'i_%s/io' % pred,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % pred,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % pred,
              'savedTestExamples':'tmp-cache/%s-test.examples' % pred,
              'learner':learner
    }

    # run the experiment
    expt.Expt(params).run()
