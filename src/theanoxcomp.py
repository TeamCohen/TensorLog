import tensorlog
import funs
import ops
import matrixdb
import declare
import mutil
import config

import theano
import theano.tensor as TT
import theano.tensor.basic as TTB
import theano.tensor.nnet as TNN
import theano.sparse as TS
import theano.sparse.basic as TSB
import scipy.sparse as SS
import numpy as NP

class TheanoEnv(object):

    """A 'namespaced' dictionary indexed by strings. Assigns every string
    to an 'internalName' which depends on the string and the
    namespaceId for this object, and indexes by that internal name.
    The internal names are assigned to keep the local variable names
    from distinct OpSeqFunction's environments from clashing, when
    they are converted to theano variables with assigned names.
    """
    def __init__(self,namespaceId):
        self.namespaceId = namespaceId
        self.env = {}
    def internalName(self,key):
        return 'n%d__%s' % (self.namespaceId,key)
    def __getitem__(self,key):
        return self.env[self.internalName(key)]
    def __setitem__(self,key,val):
        self.env[self.internalName(key)] = val

class AbstractCrossCompiler(object):
    """ Base class for tensorlog -> theano cross-compiler
    """

    def __init__(self,db):
        # namespaces are integer
        self.nameSpace = 0
        # dbMatVar is a cache mapping a mode to a theano shared
        # variable that will be bound to the (untransposed) matrix
        # returned by the matrixdb
        self.dbMatVar = {}
        # dbVecVar is a similar but for unary predicates/vectors in
        # the matrixdb
        self.dbVecVar = {}
        # dbConstVar holds constants like all-ones and one-hot vectors
        # for db constants
        self.dbConstVar = {}
        # pointer back to the matrixdb
        self.db = db
        #
        # stuff below is set by compile
        #
        # a theano expression implementing the tensorlog function, and
        # the theano variable which is the input argument(s) to
        # tensorlog function
        self.exprArgs = self.expr = None
        # theano function
        self.thFun = None

    def allocNamespace(self):
        """Allocate a new name space.
        """
        result = self.nameSpace
        self.nameSpace += 1
        return result

    def show(self):
        """ print a summary to stdout
        """
        print 'exprArgs',self.exprArgs
        print 'expr',theano.pp(self.expr)
        print 'expr.sum()',theano.pp(self.expr.sum())
        print 'debug expr.sum()\n',theano.printing.debugprint(self.expr.sum())
        print len(self.dbMatVar),'db matrices:'
        for k,v in self.dbMatVar.items():
            print ' |',k,v
        print len(self.dbVecVar),'db vectors:'
        for k,v in self.dbVecVar.items():
            print ' |',k,v
        print 'fun\n',theano.pp(self.thFun.maker.fgraph.outputs[0])
        print 'debug fun\n',theano.printing.debugprint(self.thFun.maker.fgraph.outputs[0])

    def debugExpr(self):
        AbstractCrossCompiler.debugVar(self.expr,0,maxdepth=20)

    @staticmethod
    def debugVar(v,depth=0,maxdepth=10):
      if depth>maxdepth:
        print '...'
      else:
        print '| '*(depth+1),
        print 'var: name',v.name,'type',type(v),'def',theano.pp(v)
        for a in v.get_parents():
          AbstractCrossCompiler.debugApply(a,depth=depth+1,maxdepth=maxdepth)

    @staticmethod
    def debugApply(a,depth=0,maxdepth=10):
      if depth>maxdepth:
        print '...'
      else:
        print '| '*(depth+1),
        print 'apply: ',a,'op',type(a.op),'output types',map(type,a.outputs)
        for v in a.inputs:
          AbstractCrossCompiler.debugVar(v,depth=depth+1,maxdepth=maxdepth)

    def compile(self,fun):
        """ Compile a tensorlog function to theano
        """
        (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
        self.dbArgs = []
        self.dbVals = []
        print 'self.args',self.exprArgs
        print 'self.expr',theano.pp(self.expr)
        self.thFun = theano.function(inputs=self.exprArgs, outputs=self.expr, mode='DebugMode')
        # for convenience
        return self

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################


class DenseMatDenseMsgCrossCompiler(AbstractCrossCompiler):
    """ Use theano's numpy wrappers for everything
    """

    def __init__(self,db):
        AbstractCrossCompiler.__init__(self,db)
        self.denseMsg = True
        self.denseMat = True
        # when messages are dense,
        # make sure the NULL value is small but bigger than zero,
        # which will be the default value
        # self.nullSmoothing = theano.shared(self.densifyMsg(self.db.nullMatrix(1)*1e-5), name="nullSmoothing")
        self.nullSmoothing = self.theanoSharedMsg(self.db.nullMatrix(1)*1e-5, name="nullSmoothing")

    # over-ride these to get different set of sparse/dense choices

    def densifyMat(self,m): return self._densify(m)
    def densifyMsg(self,v): return self._densify(v)
    def densifyVec(self,v): return self._densify(v)

    def sparsifyMat(self,m): return self._sparsify(m)
    def sparsifyVec(self,v): return self._sparsify(v)
    def sparsifyMsg(self,v): return self._sparsify(v)

    def _densify(self,x):
        return x.todense()
    def _sparsify(self,x):
        sx = SS.csr_matrix(x)
        sx.eliminate_zeros()
        return sx

    # over-ride these for different types of theano row variables
    def theanoSharedMat(self,val,name=None):
      #return theano.shared(self.densifyMat(val), name=name)
      x = theano.shared(self.densifyMat(val), name=name)
      print 'sharedMat',x,'type',type(x)
      return x
    def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
    def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
    def theanoRowVar(self,name): return TT.drow(name)


    #
    # the main compilation routines
    #

    def evalSymbols(self,inputSyms):
        assert len(inputSyms)==len(self.exprArgs)
        def sym2Vector(sym): return densifyMsg(self.db.onehot(sym))
        inputs = map(lambda sym:self.densifyMsg(self.db.onehot(sym)), inputSyms)
        formalArgs = inputs+self.dbVals
        theanoResult = self.thFun(*formalArgs)
        return map(lambda v:self.sparsifyMsg(v), theanoResult)

    #
    # the main compilation routines
    #

    def matrixExpr(self,matMode):
        """Return a theano expression that denotes the (untransposed) matrix
        retrieved by matrixdb
        """
        if (matMode) not in self.dbMatVar:
            u = "M__" + matMode.getFunctor() +"_" + "".join([matMode.arg(i) for i in range(matMode.getArity())])
            m = self.db.matrix(matMode,False)
            self.dbMatVar[matMode] = self.theanoSharedMat(m, name=u)
        return self.dbMatVar[matMode]

    def vectorExpr(self,matMode):
        """Return a theano expression that denotes the vector retrieved by
        the (matMode, transpose) pair using the matrixdb
        """
        assert matMode.arity==1
        if (matMode) not in self.dbVecVar:
            u = "v__" + matMode.getFunctor() +"_" + matMode.arg(0)
            v = self.db.vector(matMode)
            self.dbVecVar[matMode] = self.theanoSharedVec(v, name=u)
        return self.dbVecVar[matMode]

    def ones(self):
        """Return a theano expression that denotes an all-ones row vector
        """
        return self._msgVar('__ones',self.db.ones())

    def onehot(self,sym):
        """Return a theano expression that denotes the onehot row vector for a constant
        """
        return self._msgVar(sym,self.db.onehot(sym))

    def _msgVar(self,key,msg):
        if key not in self.dbConstVar:
            self.dbConstVar[key] = self.theanoSharedMsg(msg,name=key)
        return self.dbConstVar[key]

    def fun2Expr(self,fun,sharedInputs=None,depth=0):
        """Return a pair (inputs, expr) where binding the inputs in theano,
        and then evaluating the expression, is roughly equivalent to
        evaluating the Function fun in tensorlog.  It's only roughly
        equivalent because one also needs to bind the necessary
        variables from the matrixdb to their values.

        The sharedInputs is used if you already have theano variables
        corresponding to the inputs to this expression.  This is the case
        when you have a SumFunction: all the subexpressions share the same inputs.
        """

        if isinstance(fun,funs.SoftmaxFunction):
            # wrap inner function with softmax function
            inputs,subExpr = self.fun2Expr(fun.fun,sharedInputs,depth=depth)
            return (inputs, TNN.nnet.softmax(subExpr) + self.nullSmoothing)

        elif isinstance(fun,funs.SumFunction):
            assert(len(fun.funs)>=1)
            inputs,accum = self.fun2Expr(fun.funs[0],sharedInputs,depth=depth)
            for f in fun.funs[1:]:
                (moreInputs,addend) = self.fun2Expr(f,inputs,depth=depth)
                assert(len(moreInputs)==len(inputs))
                accum = accum+addend
            return (inputs,accum)

        elif isinstance(fun,funs.OpSeqFunction):
            assert len(fun.opInputs)==1, 'mismatching number of inputs'
            # thEnv, a 'theano environment', maps nameSpaced variables
            # from the OpSeqFunction's environment to the
            # corresponding theano subexpressions
            thEnv = TheanoEnv(self.allocNamespace())
            seqInputs = []
            if sharedInputs==None:
                # create the list of theano variables, which should be
                # used as inputs to the expression
                for v in fun.opInputs:
                    thEnv[v] = self.theanoRowVar(thEnv.internalName(v))
                    seqInputs.append(thEnv[v])
            else:
                # copy over the existing inputs to the new environment
                assert len(fun.opInputs)==len(sharedInputs)
                for i in range(len(fun.opInputs)):
                    v = fun.opInputs[i]
                    thEnv[v] = sharedInputs[i]
                    seqInputs.append(thEnv[v])
            # fill in the theano environment appropriately
            for op in fun.ops:
                thEnv[op.dst] = self.op2Expr(thEnv,op,depth)
            # return the inputs and the expression for the
            # OpSeqFunction's output
            return (seqInputs, thEnv[fun.ops[-1].dst])

        else:
            assert False,'cannot cross-compile %r' % fun

    # operator expressions for dense matrices
    def op2Expr(self,thEnv,op,depth):
        """Extend the theano environment with an expression for the
        destination of the Operator, for dense matrices
        """
        if isinstance(op,ops.VecMatMulOp):
            mExpr = self.matrixExpr(op.matMode)
            if op.transpose:
                mExpr = mExpr.T
            return thEnv[op.src].dot(mExpr)
        elif isinstance(op,ops.AssignPreimageToVar):
            mExpr = self.matrixExpr(op.matMode)
            return self.ones().dot(mExpr.T)
        elif isinstance(op,ops.ComponentwiseVecMulOp):
            return thEnv[op.src] * thEnv[op.src2]
        elif isinstance(op,ops.DefinedPredOp):
            _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
            return subExpr
        elif isinstance(op,ops.AssignOnehotToVar):
            return self.onehot(op.onehotConst)
        elif isinstance(op,ops.AssignVectorToVar):
            return self.vectorExpr(op.matMode)
        elif isinstance(op,ops.WeightedVec):
            return thEnv[op.vec] * TT.sum(thEnv[op.weighter], axis=1, keepdims=True)
        else:
            assert False,'cannot cross-compile %r' % op

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):
    """ Use theano's numpy wrappers for everything
    """

    def __init__(self,db):
        DenseMatDenseMsgCrossCompiler.__init__(self,db)
        self.denseMat = False

    # over-ride these to keep sparse matrices
    def densifyMat(self,m): return m
    def sparsifyMat(self,m): return m

    # over-ride these for different types of theano row variables
    def theanoSharedMat(self,val,name=None):
      #return theano.shared(self.densifyMat(val), name=name)
      x = theano.shared(self.densifyMat(val), name=name)
      print 'sharedMat',x,'type',type(x)
      return x
    def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
    def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
    def theanoRowVar(self,name): return TT.drow(name)

    # operator expressions for sparse matrices
    def op2Expr(self,thEnv,op,depth):
        if isinstance(op,ops.VecMatMulOp):
            mExpr = self.matrixExpr(op.matMode)
            if op.transpose:
                mExpr = mExpr.T
            return TSB.structured_dot(thEnv[op.src],mExpr)
        elif isinstance(op,ops.AssignPreimageToVar):
            mExpr = self.matrixExpr(op.matMode)
            # TODO: not sure why this simple expression doesn't work: TSB.dot(self.ones(), mExpr.transpose())
            # return TSB.dot(self.ones(), mExpr.transpose())
            return TSB.structured_dot(mExpr,self.ones().transpose()).transpose()
        elif isinstance(op,ops.ComponentwiseVecMulOp):
            return thEnv[op.src] * thEnv[op.src2]
        elif isinstance(op,ops.DefinedPredOp):
            _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
            return subExpr
        elif isinstance(op,ops.AssignOnehotToVar):
            return self.onehot(op.onehotConst)
        elif isinstance(op,ops.AssignVectorToVar):
            return self.vectorExpr(op.matMode)
        elif isinstance(op,ops.WeightedVec):
            return thEnv[op.vec] * TT.sum(thEnv[op.weighter], axis=1, keepdims=True)
        else:
            assert False,'cannot cross-compile %r' % op
