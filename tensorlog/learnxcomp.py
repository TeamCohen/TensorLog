import theanoxcomp as X
import learn as L
import theano
import theano.tensor as TT
import funs
#from termios import XCASE

#theano.config.exception_verbosity='high'

class XLearner(object):
  def __init__(self,tlprog,xc=None,compilerClass=X.DenseMatDenseMsgCrossCompiler):
    if xc: self.xc = xc
    else: self.xc = compilerClass(tlprog.db)
    self.prog = tlprog
  def predict(self,mode,X,pad=None):
    """Make predictions on a data matrix associated with the given mode."""
    #if not pad: pad = opfunutil.Scratchpad() 
    result = self.xc.eval([X])[0]
    return result
  def crossEntropy(Y,P,perExample=False):
    """Compute cross entropy some predications relative to some labels."""
    if perExample: return self.xc.dataLossFun([Y,P])
    else: assert 'No per example xe yet' #return self.xe_all([Y,P])
  def crossEntropyGrad(self,mode,X,Y,tracerArgs={},pad=None):
    """Compute the parameter gradient associated with softmax
    normalization followed by a cross-entropy cost function.  If a
    scratchpad is passed in, then intermediate results of the
    gradient computation will be saved on that scratchpad.
    """
    #if not pad: pad = opfunutil.Scratchpad()

    # More detail: in learning we use a softmax normalization
    # followed immediately by a crossEntropy loss, which has a
    # simple derivative when combined - see
    # http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
    # So in doing backprop, you don't call backprop on the outer
    # function, instead you compute the initial delta of P-Y, the
    # derivative for the loss of the (softmax o crossEntropy)
    # function, and it pass that delta down to the inner function
    # for softMax

    # do the prediction, saving intermediate outputs on the scratchpad
    predictFun = self.prog.getPredictFunction(mode)
    assert isinstance(predictFun,funs.SoftmaxFunction),'crossEntropyGrad specialized to work for softmax normalization'
    #P = self.predict(mode,X,pad)

    # compute gradient
    #paramGrads = L.GradAccumulator()
    #TODO assert rowSum(Y) = all ones - that's assumed here in
    #initial delta of Y-P
    #xc.fun.backprop(Y-P,paramGrads,pad)
    paramGrads = self.xc.evalDataLossGrad(X,Y)

    # the tracer function may output status, and may also write
    # information to the counters in paramGrads
    #self.tracer(self,paramGrads,Y,P,**tracerArgs)

    return paramGrads
        