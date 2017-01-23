import theanoxcomp as X
import learn as L
import theano
import theano.tensor as TT
import funs

#theano.config.exception_verbosity='high'

def build_crossEntropyExpr(xc):
    P=TT.dmatrix("P")
    Y=TT.dmatrix("Y")
    result = -(Y*TT.log(P)).sum()
    return theano.function(inputs=[Y,P],outputs=result/Y.shape[0]), theano.function(inputs=[Y,P],outputs=result)

class XLearner(object):
    def __init__(self,tlprog,compilerClass=X.DenseMatDenseMsgCrossCompiler):
        self.xc = compilerClass(tlprog.db)
        self.prog = tlprog
        self.xe_perExample, self.xe_all = build_crossEntropyExpr(self.xc)
        self.mode=None
        self.xeg = {}
    def _setMode(self,mode):
        if self.mode != mode:
            predictFun = self.prog.getPredictFunction(mode)
            self.xc.compile(predictFun)
            self.mode=mode
    def predict(self,mode,X,pad=None):
        """Make predictions on a data matrix associated with the given mode."""
        #if not pad: pad = opfunutil.Scratchpad() 
        self._setMode(mode)
        result = self.xc.eval([X])[0]
        return result
    def crossEntropy(Y,P,perExample=False):
        """Compute cross entropy some predications relative to some labels."""
        if perExample: return self.xe_perExample([Y,P])
        else: return self.xe_all([Y,P])
    def _setCrossEntropyGradExpr(self,mode):
        self._setMode(mode)
        if mode in self.xeg:return
        Y=TT.dmatrix("Y")
        P=self.xc.expr
        cost = Y-P
        params = self.xc.paramArgs
        paramGrads = [TT.grad(cost.sum(), param) for param in params]
        self.xeg[mode] = theano.function(inputs=self.xc.exprArgs + [Y],outputs=paramGrads),params
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
        self._setCrossEntropyGradExpr(mode)
        args = self.xc.prepare(X) + self.xc.dbArgs + self.xc.prepare(Y)
        paramGrads = self.xeg[mode][0](*args)

        # the tracer function may output status, and may also write
        # information to the counters in paramGrads
        #self.tracer(self,paramGrads,Y,P,**tracerArgs)

        return dict(zip(self.xeg[mode][1],paramGrads))
        