import os.path
import scipy.sparse as SS
import scipy.io

from tensorlog import expt
from tensorlog import comline
from tensorlog import learn
from tensorlog import ops
import logging



from tensorlog import xctargets

CROSSCOMPILERS = []
CROSSLEARNERS = {}
if xctargets.theano:
  from tensorlog import theanoxcomp
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    CROSSCOMPILERS.append(c)
    CROSSLEARNERS[c]=theanoxcomp.FixedRateGDLearner
if xctargets.tf:
  from tensorlog import tensorflowxcomp
  for c in [
    tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
    tensorflowxcomp.SparseMatDenseMsgCrossCompiler,
    ]:
    CROSSCOMPILERS.append(c)
    CROSSLEARNERS[c]=tensorflowxcomp.FixedRateGDLearner

stem = "top-1000-near-google"
        
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    ops.conf.pprintMaxdepth=50
    optdict,args = comline.parseCommandLine('--prog {stem}-recursive.ppr --proppr --db tmp-cache/{stem}.db|{stem}.cfacts:{stem}-fact.cfacts:{stem}-rule.cfacts --train tmp-cache/{stem}-train.dset|raw/{stem}.train.examples --test tmp-cache/{stem}-test.dset|raw/{stem}.test.examples'.format(stem=stem).split())
    prog = optdict['prog']
    prog.setRuleWeights()#prog.db.ones())
    prog.maxDepth=4
    params = {'prog':prog,
              #'theoryPred':'concept_atdate',
              'trainData':optdict['trainData'],
              'testData':optdict['testData'],
              'savedModel':'%s-trained.db' % stem,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % stem,
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
              'learner':learn.FixedRateGDLearner(prog,epochs=5)
    }
    expt.Expt(params).run()
    
    from tensorlog import declare
    for compilerClass in CROSSCOMPILERS:
        print compilerClass
        xc = compilerClass(prog)
        # compile everything
        #problem = declare.asMode('concept_politicianusendorsedbypoliticianus/io')
        for mode in optdict['trainData'].modesToLearn():
            xc.ensureCompiled(mode)
        learner = CROSSLEARNERS[compilerClass](prog,xc,epochs=5)
        params.update({
                #'targetMode':problem,'testData':None,
                  'savedTestPredictions':'tmp-cache/%s-test.%s.solutions.txt' % (stem,expt.fulltype(xc)),
                  'savedTestExamples':None,
                  'savedTrainExamples':None,
                  'learner':learner,
                  })
        testAcc,testXent = expt.Expt(params).run()
