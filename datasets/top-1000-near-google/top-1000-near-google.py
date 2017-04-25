import os.path
import scipy.sparse as SS
import scipy.io

from tensorlog import expt
from tensorlog import comline
from tensorlog import learn
import logging

logging.basicConfig(level=logging.INFO)

stem = "top-1000-near-google"
        
if __name__=="__main__":
    optdict,args = comline.parseCommandLine('--prog {stem}-recursive.ppr --proppr --db tmp-cache/{stem}.db|{stem}.cfacts:{stem}-fact.cfacts:{stem}-rule.cfacts --train tmp-cache/{stem}-train.dset|raw/{stem}.train.examples --test tmp-cache/{stem}-test.dset|raw/{stem}.test.examples'.format(stem=stem).split())
    prog = optdict['prog']
    prog.setRuleWeights()#prog.db.ones())
    prog.maxDepth=4
    params = {'prog':prog,
              #'theoryPred':'concept_atdate',
              'trainData':optdict['trainData'],
              'testData':optdict['testData'],
              'savedModel':'%s-trained.db' % stem,
              'savedTestPredictions':'%s-test.solutions.txt' % stem,
              'savedTrainExamples':'%s-train.examples' % stem,
              'savedTestExamples':'%s-test.examples' % stem,
              'learner':learn.FixedRateGDLearner(prog,epochs=5)
    }
    expt.Expt(params).run()
