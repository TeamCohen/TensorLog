import learn as L
import time
import sys
import logging

class XLearner(L.Learner):
  def __init__(self,prog,xc,compilerClass=None,regularizer=None,tracer=None,epochTracer=None):
    super(XLearner,self).__init__(prog,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
    if xc: self.xc = xc
    else: self.xc = compilerClass(prog.db)
  def predict(self,mode,X,pad=None):
    """Make predictions on a data matrix associated with the given mode."""
    logging.debug("XLearner predict %s"%mode)
    try:
      inferenceFun = self.xc.inferenceFunction(mode)
      result = inferenceFun(X)
    except:
      print "tlogfun:","\n".join(self.xc.ws.tensorlogFun.pprint())
      raise
    return result
  def crossEntropyGrad(self,mode,X,Y,tracerArgs={},pad=None):
    """Compute the parameter gradient associated with softmax
    normalization followed by a cross-entropy cost function.  If a
    scratchpad is passed in, then intermediate results of the
    gradient computation will be saved on that scratchpad.
    """
    gradFun = self.xc.dataLossGradFunction(mode)
    paramsWithUpdates = gradFun(X,Y)
    return paramsWithUpdates
  def applyUpdate(self, paramGrads, rate):
    assert "Cross-compilers don't apply updates"
  def meanUpdate(self, functor, arity, delta, n, totalN=0):
    assert "Cross-compilers don't do mean updates"
  def train(self,dset):
    assert False, 'abstract method called'
    
class BatchEpochsLearner(XLearner):
  def __init__(self,prog,xc,epochs=10,compilerClass=None,regularizer=None,tracer=None,epochTracer=None):
    super(BatchEpochsLearner,self).__init__(prog,xc,compilerClass=compilerClass,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
    self.epochs=epochs
  def trainMode(self,mode,X,Y,epochs=-1):
    assert False, 'abstract method called'
  def train(self,dset):
    trainStartTime = time.time()
    modes = dset.modesToLearn()
    numModes = len(modes)
    for i in range(self.epochs):
      startTime = time.time()
      for j,mode in enumerate(dset.modesToLearn()):
        args = {'i':i,'startTime':startTime,'mode':str(mode)}
        try:
          self.trainMode(mode,dset.getX(mode),dset.getY(mode),epochs=1)
        except:
          print "Unexpected error at %s:" % str(args), sys.exc_info()[:2]
          raise
      #self.epochTracer(self,)
    
  
        
