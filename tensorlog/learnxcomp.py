import theanoxcomp as X
import learn as L
import theano
import theano.tensor as TT
import funs
#from termios import XCASE

#theano.config.exception_verbosity='high'

class XLearner(L.Learner):
  def __init__(self,prog,xc=None,compilerClass=X.DenseMatDenseMsgCrossCompiler,regularizer=None,tracer=None,epochTracer=None):
    super(XLearner,self).__init__(prog,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
    if xc: self.xc = xc
    else: self.xc = compilerClass(prog.db)
  def predict(self,mode,X,pad=None):
    """Make predictions on a data matrix associated with the given mode."""
    #if not pad: pad = opfunutil.Scratchpad() 
    result = self.xc.eval([X])[0]
    return result
  def crossEntropy(self,Y,P,perExample=False):
    """Compute cross entropy some predications relative to some labels."""
    if perExample: return self.xc.dataLossFun([Y,P])
    else: assert 'No per example xe yet' #return self.xe_all([Y,P])
  def crossEntropyGrad(self,mode,X,Y,tracerArgs={},pad=None):
    """Compute the parameter gradient associated with softmax
    normalization followed by a cross-entropy cost function.  If a
    scratchpad is passed in, then intermediate results of the
    gradient computation will be saved on that scratchpad.
    """
    paramGrads = self.xc.evalDataLossGrad(X,Y)
    return paramGrads
  def applyUpdate(self, paramGrads, rate):
    assert "Cross-compilers don't apply updates"
  def meanUpdate(self, functor, arity, delta, n, totalN=0):
    assert "Cross-compilers don't do mean updates"
  
        