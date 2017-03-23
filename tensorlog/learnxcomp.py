import learn as L

class XLearner(L.Learner):
  def __init__(self,prog,xc,compilerClass=None,regularizer=None,tracer=None,epochTracer=None):
    super(XLearner,self).__init__(prog,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
    if xc: self.xc = xc
    else: self.xc = compilerClass(prog.db)
  def predict(self,mode,X,pad=None):
    """Make predictions on a data matrix associated with the given mode."""
    inferenceFun = self.xc.inferenceFunction(mode)
    result = inferenceFun(X)
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
  
        