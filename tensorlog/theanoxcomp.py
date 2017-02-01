import theano
import theano.tensor as TT
import theano.tensor.basic as TTB
import theano.tensor.nnet as TNN
import theano.sparse as TS
import theano.sparse.basic as TSB
import theano.sparse.type as TST
import scipy.sparse as SS
import numpy as NP

from tensorlog import funs
from tensorlog import ops

from tensorlog import xcomp

class TheanoCrossCompiler(xcomp.AbstractCrossCompiler):

  def compile(self,fun,params=None):
    """Compile a tensorlog function to theano.  Params are optional, if
    they are given then also compile gradient of the loss function.
    Params should be a list of (functor,arity) pairs.
    """
    (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
    # can also add mode='DebugMode'
    self.inferenceFun = theano.function(inputs=(self.exprArgs + self._secondaryArgs()),
                                        outputs=self.expr)
    self._buildLossExpr(params)
    return self

  def _buildLossExpr(self,params):
    """ Add in the stuff relating to loss"""
    target_y = self._vectorVar('_target_y')
    self.dataTargetArgs = [target_y]
    #this expr is safe for vectors with zeros - TNN.nnet.categorical_crossentropy blows up
    #self.dataLossExpr = (-target_y*TT.log(self.expr+1)).mean()
    # get the non-zero values of the dense expression
    self.dataLossExpr = (target_y*self.applyOpToNonzerosOfDense(TT.log,self.expr)).mean()
    self.dataLossFun = theano.function(
        inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
        outputs=(self.expr,self.dataLossExpr))
    if params is not None:
      self.params = params
      paramVars = map(self.getSubExpr,params)
      self.dataLossGradExprs = theano.grad(self.dataLossExpr, paramVars)
      self.dataLossGradFun = theano.function(
          inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
          outputs=self.dataLossGradExprs)

  #
  # evaluators
  #

  def wrapSymbols(self,inputSyms):
    """ Convert a list of symbols to a list of one-hot vectors that can be sent to eval"""
    return map(lambda sym:self._wrapDBVector(self.db.onehot(sym)), inputSyms)

  def _unwrapOutputs(self,targetFunctionOutputs):
    return map(lambda v:self._sparsify(v), targetFunctionOutputs)

  def eval(self,inputs):
    formalArgs = inputs + self._secondaryArgBindings()
    assert len(formalArgs)>0
    return self._unwrapOutputs(self.inferenceFun(*formalArgs))

  def evalDataLoss(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self._wrapDBVector, rawInputs)
    target = self._wrapDBVector(rawTarget)
    formalArgs = inputs + [target] + self._secondaryArgBindings()
    return self._unwrapOutputs(self.dataLossFun(*formalArgs))

  def evalDataLossGrad(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self._wrapDBVector, rawInputs)
    target = self._wrapDBVector(rawTarget)
    formalArgs = inputs + [target] + self._secondaryArgBindings()
    return self._unwrapOutputs(self.dataLossGradFun(*formalArgs))

  def applyOpToNonzerosOfDense(self,op,expr):
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)

  @staticmethod
  def theanoTransposer(mx): return mx.T

  # for debugging output

  def show(self):
    """ print a summary to stdout """
    print 'exprArgs',self.exprArgs
    print 'expr',theano.pp(self.expr)
    print 'debug expr.sum()\n',theano.printing.debugprint(self.expr.sum())
    print 'subexpr cache:'
    for k,v in self.subexprCacheVarBindings.items():
      print ' |',k,'type',type(v)
    print 'function:',theano.pp(self.inferenceFun.maker.fgraph.outputs[0])
    #print 'debug fun\n',theano.printing.debugprint(self.inferenceFun.maker.fgraph.outputs[0])


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

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################


class DenseMatDenseMsgCrossCompiler(TheanoCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def _vectorVar(self,name):
    return TT.drow(name)

  def _matrixVar(self,name):
    return TT.dmatrix(name)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat.todense()

  def _sparsify(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx


  #
  # the main compilation routines
  #

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
      softmaxOverNonzeros = self.applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self.nullSmoothing)
      return (inputs, softmaxOverNonzeros)

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
      thEnv = xcomp.NameSpacer(self.allocNamespace())
      seqInputs = []
      if sharedInputs==None:
        # create the list of theano variables, which should be
        # used as inputs to the expression
        for v in fun.opInputs:
          thEnv[v] = self._vectorVar(thEnv.internalName(v))
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

    elif isinstance(fun,funs.NullFunction):
      return ([], self.zeros())

    else:
      assert False,'cannot cross-compile %r' % fun

  # operator expressions for dense matrices
  def op2Expr(self,thEnv,op,depth):
    """Extend the theano environment with an expression for the
    destination of the Operator, for dense matrices
    """
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose,TheanoCrossCompiler.theanoTransposer)
      return thEnv[op.src].dot(mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode,False,TheanoCrossCompiler.theanoTransposer)
      return self.ones().dot(mExpr.T)
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return thEnv[op.src] * thEnv[op.src2]
    elif isinstance(op,ops.DefinedPredOp):
      _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self.onehot(op.onehotConst)
    elif isinstance(op,ops.AssignVectorToVar):
      return self.vector(op.matMode)
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

  def _vectorVar(self,name):
    return TT.drow(name)

  def _matrixVar(self,name):
    return TSB.matrix('csr',name=name)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat

  def _sparsify(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  # operator expressions for sparse matrices
  def op2Expr(self,thEnv,op,depth):
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose,TheanoCrossCompiler.theanoTransposer)
      return TSB.structured_dot(thEnv[op.src],mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode,False,TheanoCrossCompiler.theanoTransposer)
      return TSB.dot(self.ones(),mExpr.T)
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return thEnv[op.src] * thEnv[op.src2]
    elif isinstance(op,ops.DefinedPredOp):
      _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self.onehot(op.onehotConst)
    elif isinstance(op,ops.AssignVectorToVar):
      return self.vector(op.matMode)
    elif isinstance(op,ops.WeightedVec):
      return thEnv[op.vec] * TT.sum(thEnv[op.weighter], axis=1, keepdims=True)
    else:
      assert False,'cannot cross-compile %r' % op
