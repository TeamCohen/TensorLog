import tensorlog
import funs
import ops
import matrixdb
import declare

import theano
import theano.tensor as TT
import theano.tensor.nnet as TNN
import theano.sparse as TS
import theano.sparse.basic as TSB
from theano.ifelse import ifelse
import numpy as NP


class CrossCompiler(object):

    def __init__(self,db):
        self.nameSpace = 0
        self.matrixEnv = {}
        self.matrixArgs = []
        self.matrixArgNames = []
        self.db = db

    def localName(self,s,ns):
        return 'n%d__%s' % (ns,s)

    def nextNameSpace(self):
        result = self.nameSpace
        self.nameSpace += 1
        return result

    def matrixFor(self,matMode,transpose):
        if ((matMode,transpose)) not in self.matrixEnv:
            u = "M__" + matMode.getFunctor() +"_" \
                + "".join([matMode.arg(i) for i in range(matMode.getArity())]) \
                + ("_T" if transpose else "")
            self.matrixEnv[u] = TT.dmatrix(u)
            self.matrixArgNames.append(u)
            self.matrixArgs.append(self.db.matrix(matMode,transpose))
        return self.matrixEnv[u]

    def fun2Expr(self,fun,numInputs):

        if isinstance(fun,funs.SoftmaxFunction):
            # wrap inner function with softmax function
            print "softMax..."
            inputs,subExpr = self.fun2Expr(fun.fun,numInputs)
            # tricky: in our softmax code, we mask the results to keep 0 entries 0
            # before normalizing :-/
            result = TNN.nnet.softmax(subExpr) * (1 - TT.isclose(subExpr,TT.zeros_like(subExpr)))
            return (inputs, result / result.sum())

        elif isinstance(fun,funs.OpSeqFunction):
            assert len(fun.opInputs)==numInputs, 'mismatching number of inputs'
            # env maps nameSpaced variables to theano subexpressions
            print "opSeq..."
            ns = self.nextNameSpace()
            thEnv = {}
            # seqInputs is a list of theano variables which can be
            # used as arguments to a function
            seqInputs = []
            for v in fun.opInputs:
                u = self.localName(v,ns)
                thEnv[u] = TT.dvector(u) #TT.dmatrix(u)
                seqInputs.append(thEnv[u])

            for op in fun.ops:
                u = self.localName(op.dst,ns)
                thEnv[u] = self.op2Expr(thEnv,ns,op)

            uOut = self.localName(op.dst,ns)
            return (seqInputs, thEnv[uOut])
        
        else:
            assert False,'cannot cross-compile %r' % fun
    
    def op2Expr(self,thEnv,ns,op):
        
        if isinstance(op,ops.VecMatMulOp):
            print "+vecMatMul..."
            u = self.localName(op.src,ns)
            return theano.dot(thEnv[u], self.matrixFor(op.matMode,op.transpose))

        else:
            assert False,'cannot cross-compile %r' % op
