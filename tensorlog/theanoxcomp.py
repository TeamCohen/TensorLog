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

  def finalizeInference(self):
    self.ws.inferenceFun = theano.function(
        inputs=self.ws.inferenceArgs,
        outputs=self.ws.inferenceExpr)

  def buildLossExpr(self,params):
    target_y = self.createVectorPlaceholder(xcomp.TRAINING_TARGET_VARNAME)
    # get the non-zero values of the dense expression
    self.ws.dataLossExpr = (target_y * self._applyOpToNonzerosOfDense(TT.log,self.ws.inferenceExpr)).mean()
    self.ws.dataLossFun = theano.function(
        inputs=(self.ws.inferenceArgs + self.dataTargetArgs),
        outputs=(self.ws.inferenceExpr, self.dataLossExpr))
    if params is not None:
      self.ws.params = params
      paramVars = map(lambda p:self.ws[p], params)
      self.ws.dataLossGradExprs = theano.grad(self.dataLossExpr, paramVars)
      self.ws.dataLossGradFun = theano.function(
          inputs=(self.ws.inferenceArgs + self.dataTargetArgs),
          outputs=self.dataLossGradExprs)

  #
  # evaluators
  #

  def eval(self,inputs):
    return self.unwrapOutputs(self.ws.inferenceFun(*inputs))

  def evalDataLoss(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self.wrapDBVector, rawInputs)
    target = self.wrapDBVector(rawTarget)
    formalArgs = inputs + [target]
    return self.unwrapOutputs(self.dataLossFun(*formalArgs))

  def evalDataLossGrad(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self.wrapDBVector, rawInputs)
    target = self.wrapDBVector(rawTarget)
    formalArgs = inputs + [target]
    return self.unwrapOutputs(self.dataLossGradFun(*formalArgs))

  def _applyOpToNonzerosOfDense(self,op,expr):
    # useful subroutine
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)

  # for debugging output

  def show(self):
    """ print a summary to stdout """
    print 'inferenceArgs',self.ws.inferenceArgs
    print 'inferenceExpr',theano.pp(self.ws.inferenceExpr)
    print 'inferenceExpr:',theano.printing.debugprint(self.ws.inferenceExpr)
    #print 'function:',theano.pp(self.ws.inferenceFun.maker.fgraph.outputs[0])
#    print 'subexpr cache:'
#    for k,v in self.subexprCacheVarBindings.items():
#      print ' |',k,'type',type(v)


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

  def createPlaceholder(self,name,kind):
    assert kind=='vector'
    return TT.drow(name)

  def createSharedVar(self,name,val,kind):
    return theano.shared(val, name=name)

  def wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat.todense()

  def unwrapOutput(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  def softmaxFun2Expr(self,fun,sharedInputs,depth):
    inputs,subExpr = self.fun2Expr(fun.fun,sharedInputs,depth)
    softmaxOverNonzeros = self._applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self.nullSmoothing)
    return (inputs, softmaxOverNonzeros)

  def transposeMatrixExpr(self,mx):
    return mx.T

  def vecMatMulExpr(self,v,m):
    return v.dot(m)

  def componentwiseMulExpr(self,v1,v2):
    return v1*v2

  def weightedVecExpr(self,vec,weighter):
    return vec * TT.sum(weighter, axis=1, keepdims=True)

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):
  """ Use theano's numpy wrappers for everything
  """

  def _vectorVar(self,name):
    return TT.drow(name)

  def createMatrixVar(self,name):
    return TSB.matrix('csr',name=name)

  def wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def wrapDBMatrix(self,mat):
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
