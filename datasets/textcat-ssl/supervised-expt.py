import sys

from tensorlog import expt
from tensorlog import declare
from tensorlog import program
from tensorlog import comline
from tensorlog import learn
from tensorlog import plearn

if __name__=="__main__":
    #usage: [stem] [epochs]

    #get the command-line options for this experiment
    stem = 'citeseer' if len(sys.argv)<=1 else sys.argv[1]
    epochs = 10 if len(sys.argv)<=2 else int(sys.argv[2])

    # use comline.parseCommandLine to set up the program, etc
    optdict,args = comline.parseCommandLine([
            '--logging', 'info',
            '--db', 'inputs/%s.db|inputs/%s-corpus.cfacts' % (stem,stem),
            '--prog','inputs/%s-textcat.ppr' % stem, '--proppr',
            '--train','inputs/%s-train.dset|inputs/%s-train.exam' % (stem,stem),
            '--test', 'inputs/%s-test.dset|inputs/%s-test.exam' % (stem,stem)])

    # prog is shortcut to the output optdict, for convenience.
    prog = optdict['prog']

    # the weight vector is sparse - just the constants in the unary predicate rule
    prog.setFeatureWeights()

    # use a non-default learner, overriding the tracing function,
    # number of epochs, and regularizer
#    learner = learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),traceFun=learn.Learner.cheapTraceFun,epochs=epochs)
#    learner = plearn.ParallelFixedRateGDLearner(prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
    learner = plearn.ParallelFixedRateGDLearner(prog,epochs=epochs,parallel=55,rate=20.0, regularizer=learn.L2Regularizer(0.0001))
#    learner = plearn.ParallelAdaGradLearner(prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
#    learner = plearn.ParallelFixedRateGDLearner(prog,epochs=epochs,parallel=40)

    # configute the experiment
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % stem,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
              'savedModel':'tmp-cache/%s-model.examples' % stem,
              'learner':learner
    }

    # run the experiment
    expt.Expt(params).run()
