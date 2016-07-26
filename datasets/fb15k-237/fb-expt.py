import sys

import matrixdb
import expt
import dataset
import declare
import tensorlog
import learn
import plearn

if __name__=="__main__":
    #usage: fb-expt.py [processes] [epochs] [targetMode] 
    
    processes = int(sys.argv[1]) if len(sys.argv)>1 else 40
    epochs = int(sys.argv[2]) if len(sys.argv)>2 else 30
    targetMode = sys.argv[3] if len(sys.argv)>3 else None
    if targetMode and (not targetMode.endswith("(i,o)")): 
        targetMode += "(i,o)"

    optdict,args = expt.Expt.timeAction(
        'parsing command line and loading data',
        lambda:tensorlog.parseCommandLine([
            '--logging','info',
            '--db', 'inputs/fb.db|inputs/fb.cfacts',
            '--prog','inputs/learned-rules.ppr', '--proppr',
            '--train','inputs/train.dset|inputs/train.exam',
            '--test', 'inputs/valid.dset|inputs/valid.exam'
        ]))

    prog = optdict['prog']
    prog.setRuleWeights(weights=prog.db.vector(declare.asMode("ruleid(i)")))
    if processes==0:
        learner = learn.FixedRateGDLearner(
            prog,
            epochs=epochs,
            #        regularizer=learn.L2Regularizer(),
        )
    else:
        learner = plearn.ParallelFixedRateGDLearner(
            prog,
            epochs=epochs,
            parallel=processes,
            miniBatchSize=100,
            #epochTracer=learn.EpochTracer.defaultPlusAcc,
            epochTracer=learn.EpochTracer.cheap,
            regularizer=learn.L2Regularizer(),
        )

    # configure the experiment
    params = {'prog':prog,
              'trainData':optdict['trainData'], 
              'testData':optdict['testData'],
              'savedModel':'tmp-cache/trained.db',
              'savedTestPredictions':'tmp-cache/valid.solutions.txt',
              'savedTestExamples':'tmp-cache/valid.examples',
              'targetMode':targetMode,
              'learner':learner
    }

    # run the experiment
    expt.Expt(params).run()

