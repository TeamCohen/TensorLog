import sys

import expt
import declare
import tensorlog
import learn
import plearn



if __name__=="__main__":
    #usage: [dataset] [epochs]
    
    #get the command-line options for this experiment
    dataset = 'yago2-sample' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 30 if len(sys.argv)<=2 else int(sys.argv[2])

    # use tensorlog.parseCommandLine to set up the program, etc
    optdict,args = tensorlog.parseCommandLine([
            '--logging', 'debug', # was: 'warn'
            '--db', 'inputs/{0}.db|inputs/{0}.cfacts'.format(dataset),
            '--prog','inputs/{0}.ppr'.format(dataset), '--proppr',
            '--train','inputs/{0}-train.dset|inputs/{0}-train.exam'.format(dataset),
            '--test', 'inputs/{0}-test.dset|inputs/{0}-test.exam'.format(dataset)])

    # prog is shortcut to the output optdict, for convenience.
    prog = optdict['prog']

    # the weight vector is sparse - just the constants in the unary predicate rule
    prog.setRuleWeights(prog.db.vector(declare.asMode("rule(i)")))
    
    # set the max recursion depth
    prog.maxDepth = 1

    # use a non-default learner, overriding the tracing function,
    # number of epochs, and regularizer
    #learner = plearn.ParallelFixedRateGDLearner(
    #    prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
    learner = learn.FixedRateGDLearner(
        prog,epochs=epochs,regularizer=learn.L2Regularizer())
#    learner = plearn.ParallelAdaGradLearner(
#        prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
    targetMode = None

    # configute the experiment
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'targetMode':targetMode,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % dataset,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % dataset,
              'savedTestExamples':'tmp-cache/%s-test.examples' % dataset,
              'learner':learner
    }

    # run the experiment
    expt.Expt(params).run()
