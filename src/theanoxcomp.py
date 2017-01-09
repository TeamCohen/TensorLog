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
        # dbMatVar is a cache mapping a (mode,transposeFlag) pair
        # -- which determine a matrix from the matrixdb -- to a theano
        # shared variable that will be bound to that
        # matrix. dbConstVar is similar but for message constants (eg,
        # onehot values)
        self.dbMatVar = {}
        self.dbVecVar = {}
        self.dbMsgVar = {}
        # hold the database
        self.db = db
        #
        # stuff below is set by compile
        #
        # a theano expression implementing the tensorlog function, and
        # the theano variable which is the input argument(s) to
        # tensorlog function
        self.exprArgs = self.expr = None
        # the theano variables for the matrixdb matrices used in the
        # expression, and the values to which they should be bound
        self.dbArgs = self.dbVals = None
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
        print len(self.dbArgs),'db param(s):'
        for i in range(len(self.dbArgs)):
            print ' |',self.dbArgs[i],type(self.dbVals[i])
        print 'fun',theano.pp(self.thFun.maker.fgraph.outputs[0])
        print 'debug fun',theano.printing.debugprint(self.thFun.maker.fgraph.outputs[0])

    def compile(self,fun):
        """ Compile a tensorlog function to theano
        """
        (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
        self.dbArgs = []
        self.dbVals = []
        print 'self.exprArgs',self.exprArgs
        print 'self.expr',theano.pp(self.expr)
        self.thFun = theano.function(inputs=self.exprArgs, outputs=self.expr)
        self.paramArgs = []
        self.paramVals = []
        for (mode,thvar) in self.dbMatVar.items():
            if self.db.isParameter(mode):
                self.paramArgs.append(thvar)
                self.paramVals.append( (mode.functor,mode.arity) )
        print "post-compile show():"
        self.show()
        # for convenience
        return self
    def evalSymbols(self,inputSyms):
        assert len(inputSyms)==len(self.exprArgs)
        #def sym2Vector(sym): return densifyMsg(self.db.onehot(sym))
        inputs = map(lambda sym:self.db.onehot(sym), inputSyms)
        #print "inputs:\n","\n".join([str(self.db.matrixAsSymbolDict(SS.csr_matrix(x))) for x in inputs])
        return self.eval(inputs)

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
        #self.nullSmoothing = self.theanoSharedMsg(self.db.nullMatrix(1)*-10, name="nullSmoothing")
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
    def theanoSharedMat(self,val,name=None): return theano.shared(self.densifyMat(val), name=name)
    def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
    def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
    def theanoRowVar(self,name): return TT.drow(name)


    #
    # the main compilation routines
    # 
    def prepare(self,inputs):
        return map(lambda onehot:self.densifyMsg(onehot),inputs)
    def eval(self,inputs):
        formalArgs = self.prepare(inputs)+self.dbVals
        theanoResult = self.thFun(*formalArgs)
        result = map(lambda v:self.sparsifyMsg(v), theanoResult)
        #print "result:\n","\n".join([str(self.db.matrixAsSymbolDict(SS.csr_matrix(x))) for x in theanoResult])
        return result

    #
    # the main compilation routines
    # 

    def matrixExpr(self,matMode):
        """Return a theano expression that denotes the matrix retrieved by
        the (matMode, transpose) pair using the matrixdb
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
        if key not in self.dbMsgVar:
            self.dbMsgVar[key] = self.theanoSharedMsg(msg,name=key)
        return self.dbMsgVar[key]

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
            # alternate (slow) solution to mimic mutil.denseSoftmax():
            #augmented = subExpr + self.nullSmoothing
            # run softmax, then filter to keep the zero inputs as zeros in the output
            # ... isclose() isn't real fast for this but it's not as slow as switch()
            #result = TNN.nnet.softmax(augmented) * (1 - TT.isclose(augmented,TT.zeros_like(augmented)))
            # renormalize after filtering
            #return (inputs, result / result.sum())

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
            print "opseq environment:",thEnv.env
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
            print 'weighted vec operator'
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
    def theanoSharedMat(self,val,name=None): return theano.sparse.shared(self.densifyMat(val), name=name, format='csr')
    def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
    def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
    def theanoRowVar(self,name): return TT.drow(name)

    # operator expressions for sparse matrices
    def op2Expr(self,thEnv,op,depth):
        if isinstance(op,ops.VecMatMulOp):
            mExpr = self.matrixExpr(op.matMode)
            if op.transpose:
                mExpr = mExpr.get_value().transpose()
            return TSB.dot(thEnv[op.src],mExpr)
        elif isinstance(op,ops.AssignPreimageToVar):
            mExpr = self.matrixExpr(op.matMode)
            # TODO: not sure why this simple expression doesn't work: TSB.dot(self.ones(), mExpr.transpose())
            return TSB.dot(mExpr,self.ones().transpose()).transpose()
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
