import logging

from tensorlog import masterconfig
from tensorlog import expt
from tensorlog import learn
from tensorlog import plearn
from tensorlog import comline
from tensorlog import xctargets

CROSSCOMPILERS = []
if xctargets.theano:
  from tensorlog import theanoxcomp
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    CROSSCOMPILERS.append(c)
    
modeString = 'answer/io'
def setExptParams(num):
    db = comline.parseDBSpec('tmp-cache/train-%d.db|inputs/train-%d.cfacts' % (num,num))
    trainData = comline.parseDatasetSpec('tmp-cache/train-%d.dset|inputs/train-%d.exam'  % (num,num), db)
    testData = comline.parseDatasetSpec('tmp-cache/test-%d.dset|inputs/test-%d.exam'  % (num,num), db)
    prog = comline.parseProgSpec("theory.ppr",db,proppr=True)
    prog.setFeatureWeights()
    learner = plearn.ParallelFixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),parallel=5,epochs=10)
    return {'prog':prog,
            'trainData':trainData,
            'testData':testData,
            'targetMode':modeString,
            'savedModel':'learned-model.db',
            'learner':learner
    }

def runMain(num=250):
    logging.basicConfig(level=logging.INFO)
    masterconfig.masterConfig().matrixdb.allow_weighted_tuples=False
    params = setExptParams(num)
    return expt.Expt(params).run()

if __name__=="__main__":
  acc,loss = runMain() # expect 0.21,0.22
  print 'acc,loss',acc,loss
  params = setExptParams(250)
  for compilerClass in CROSSCOMPILERS:
      start0=time.time()
      xc = compilerClass(ti.prog)
      print expt.fulltype(xc)
      # compile everything
      mode = declare.asMode(modeString)
      xc.ensureCompiled(mode)
      print 'eval',modeString,
      start = time.time()
      xc.inferenceFunction(mode)(X)
      print 'total time',expt.fulltype(xc),time.time()-start0,'sec'
