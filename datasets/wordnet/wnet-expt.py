import sys

import expt
import declare
import tensorlog
import learn
import plearn

if __name__=="__main__":
    #usage: [targetPredicate] [epochs]
    
    #get the command-line options for this experiment
    pred = 'ALL' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 30 if len(sys.argv)<=2 else int(sys.argv[2])

    # use tensorlog.parseCommandLine to set up the program, etc
    optdict,args = tensorlog.parseCommandLine([
            '--logging', 'warn',
            '--db', 'inputs/wnet.db|inputs/wnet.cfacts',
            '--prog','inputs/wnet-learned.ppr', '--proppr',
            '--train','inputs/wnet-train.dset|inputs/wnet-train.exam',
            '--test', 'inputs/wnet-test.dset|inputs/wnet-valid.exam'])

    # prog is shortcut to the output optdict, for convenience.
    prog = optdict['prog']

    # the weight vector is sparse - just the constants in the unary predicate rule
    prog.setWeights(prog.db.vector(declare.asMode("rule(i)")))

    # use a non-default learner, overriding the tracing function,
    # number of epochs, and regularizer
    learner = plearn.ParallelFixedRateGDLearner(
        prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
    targetMode = 'i_%s/io' % pred if pred!='ALL' else None

    # configute the experiment
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'targetMode':targetMode,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % pred,
              'savedTrainExamples':'tmp-cache/wnet-train.examples',
              'savedTestExamples':'tmp-cache/wnet-test.examples',
              'learner':learner
    }

    # run the experiment
    expt.Expt(params).run()
