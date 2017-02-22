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
    target_y = self.createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self.ws.inferenceOutputType)
    self.ws.dataLossArgs = [target_y]
    self.ws.dataLossExpr = (-target_y * self._applyOpToNonzerosOfDense(TT.log,self.ws.inferenceExpr)).mean()
    if params is not None:
      self.ws.params = params
      paramVars = map(lambda p:self.ws.getHandleExprVariable(p), params)
      self.ws.dataLossGradExprs = theano.grad(self.ws.dataLossExpr, paramVars)
    #finalize
    self.ws.dataLossFun = theano.function(
        inputs=(self.ws.inferenceArgs + self.ws.dataLossArgs),
        outputs=self.ws.dataLossExpr)
    if params is not None:
      self.ws.dataLossGradFun = theano.function(
          inputs=(self.ws.inferenceArgs + self.ws.dataLossArgs),
          outputs=self.ws.dataLossGradExprs)

  #
  # evaluators
  #

  def eval(self,rawInputs):
    inputs = map(self.wrapMsg,rawInputs)
    return self.unwrapOutputs(self.ws.inferenceFun(*inputs))

  def evalDataLoss(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self.wrapMsg, rawInputs)
    target = self.wrapMsg(rawTarget)
    formalArgs = inputs + [target]
    return self.unwrapOutput(self.ws.dataLossFun(*formalArgs))

  def evalDataLossGrad(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self.wrapMsg, rawInputs)
    target = self.wrapMsg(rawTarget)
    formalArgs = inputs + [target]
    rawUpdates = self.ws.dataLossGradFun(*formalArgs)
    return map(lambda key,rawUpdate:self.unwrapUpdate(key,rawUpdate), self.ws.params, rawUpdates)

  def _applyOpToNonzerosOfDense(self,op,expr):
    # useful subroutine
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)

  def show(self,verbose=0):
    """ print a summary to stdout """
    print 'inferenceArgs',self.ws.inferenceArgs
    print 'inferenceExpr',theano.pp(self.ws.inferenceExpr)
    if verbose>=1:
      print 'debugprint inferenceExpr:'
      theano.printing.debugprint(self.ws.inferenceExpr)
      if self.ws.dataLossExpr:
        print 'dataLossArgs',self.ws.dataLossArgs
        print 'dataLossExpr',theano.pp(self.ws.dataLossExpr)
        print 'debugprint dataLossExpr:'
        theano.printing.debugprint(self.ws.dataLossExpr)

  def insertHandleExpr(self,key,name,val):
    self.ws._handleExpr[key] = self.ws._handleExprVar[key] = theano.shared(val, name=name)

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TheanoCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def createPlaceholder(self,name,kind,typeName):
    assert kind=='vector'
    result = TT.drow(name)
    return result

  def wrapMsg(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

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

  def unwrapUpdate(self,key,up):
    return self.unwrapOutput(up)

  def softmaxFun2Expr(self,subExpr,typeName):
    return self._applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self.nullSmoother[typeName])

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

  def wrapDBMatrix(self,mat):
    return mat

  def vecMatMulExpr(self,v,m):
    return TSB.structured_dot(v,m)
