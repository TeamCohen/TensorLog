import os.path
import sys
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

stem = "top-1000-near-google"

def setExptParams():
  if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
  db = comline.parseDBSpec('tmp-cache/{stem}.db|{stem}.cfacts:{stem}-fact.cfacts:{stem}-rule.cfacts'.format(stem=stem))
  trainData = comline.parseDatasetSpec('tmp-cache/{stem}-train.dset|raw/{stem}.train.examples'.format(stem=stem), db)
  testData = comline.parseDatasetSpec('tmp-cache/{stem}-test.dset|raw/{stem}.test.examples'.format(stem=stem), db)
  prog = comline.parseProgSpec("{stem}-recursive.ppr".format(stem=stem),db,proppr=True)
  prog.setRuleWeights()
  prog.maxDepth=4
  learner = learn.FixedRateGDLearner(prog,epochs=5)
  return {'prog':prog,
          'trainData':trainData,
          'testData':testData,
          'savedModel':'%s-trained.db' % stem,
          'savedTestPredictions':'tmp-cache/%s-test.solutions.txt' % stem,
          'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
          'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
          'learner':learner
        }

def runMain(cross=False):
    logging.basicConfig(level=logging.INFO)
    params = setExptParams()
    result=[]
    result.append(expt.Expt(params).run())
    if cross:
      for compilerClass in CROSSCOMPILERS:
        xc = compilerClass(prog)
        print(expt.fulltype(xc))
        
        # compile everything
        #problem = declare.asMode('concept_politicianusendorsedbypoliticianus/io')
        for mode in params['trainData'].modesToLearn():
            xc.ensureCompiled(mode)
        learner = CROSSLEARNERS[compilerClass](prog,xc,epochs=5)
        params.update({
                #'targetMode':problem,'testData':None,
                  'savedTestPredictions':'tmp-cache/%s-test.%s.solutions.txt' % (stem,expt.fulltype(xc)),
                  'savedTestExamples':None,
                  'savedTrainExamples':None,
                  'learner':learner,
                  })
        result.append( (expt.fulltype(xc),expt.Expt(params).run()) )
    return result
                  

if __name__=="__main__":
  acc,loss = runMain('cross' in sys.argv[1:])[0]
  print('acc,loss',acc,loss)
