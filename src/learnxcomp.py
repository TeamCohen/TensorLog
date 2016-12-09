import theanoxcomp as X
import learn as L
import theano
import theano.tensor as TT

def build_crossEntropyExpr(xc):
    P=xc.theanoMatrix("P")
    Y=xc.theanoMatrix("Y")
    result = -(Y*TT.log(P)).sum()
    return theano.function(inputs=[Y,P],outputs=result/Y.shape[0]),theano.function(inputs=[Y,P],outputs=result)

class XLearner(object):
    def __init__(self,tlprog,compilerClass=X.DenseMatDenseMsgCrossCompiler):
        self.xc = compilerClass(tlprog.db)
        self.prog = tlprog
        self.xe_perExample, self.xe_all = build_crossEntropyExpr(self.xc)  
    def predict(self,mode,X,pad=None):
        """Make predictions on a data matrix associated with the given mode."""
        #if not pad: pad = opfunutil.Scratchpad() 
        predictFun = self.prog.getPredictFunction(mode)
        self.xc.compile(predictFun)
        result = self.xc.eval([X])[0]
        return result
    def crossEntropy(Y,P,perExample=False):
        """Compute cross entropy some predications relative to some labels."""
        if perExample: return self.xe_perExample([Y,P])
        else: return self.xe_all([Y,P])
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
        P = self.predict(mode,X,pad)

        # compute gradient
        #paramGrads = L.GradAccumulator()
        #TODO assert rowSum(Y) = all ones - that's assumed here in
        #initial delta of Y-P
        cost = Y-P
        #params = 
        xc.fun.backprop(Y-P,paramGrads,pad)

        # the tracer function may output status, and may also write
        # information to the counters in paramGrads
        #self.tracer(self,paramGrads,Y,P,**tracerArgs)

        return paramGrads
        