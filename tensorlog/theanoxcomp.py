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

  def _buildLossExpr(self):
    target_y = self._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self.ws.inferenceOutputType)
    self.ws.dataLossArgs = self.ws.inferenceArgs + [target_y]
    self.ws.dataLossExpr = (-target_y * self._applyOpToNonzerosOfDense(TT.log,self.ws.inferenceExpr)).mean()
    # need to keep this in a canonical order
    self.ws.params = list(self.db.params)
    paramVars = map(lambda p:self.ws._getHandleExprVariable(p), self.ws.params)
    self.ws.dataLossGradExprs = theano.grad(self.ws.dataLossExpr, paramVars)
    #finalize
#    self.ws.dataLossFun = theano.function(
#        inputs=(self.ws.inferenceArgs + self.ws.dataLossArgs),
#        outputs=self.ws.dataLossExpr)
#    if self.ws.params:
#      self.ws.dataLossGradFun = theano.function(
#          inputs=(self.ws.inferenceArgs + self.ws.dataLossArgs),
#          outputs=self.ws.dataLossGradExprs)

  def _asFunction(self,args,expr,wrapInputs,unwrapOutputs):
    pyfun = theano.function(inputs=args, outputs=expr)
    def closure(rawInputs):
       inputs = map(self._wrapMsg,rawInputs) if wrapInputs else rawInputs
       tmp = pyfun(*inputs)[0]
       return self._unwrapOutput(tmp) if unwrapOutputs else tmp
    return closure

  def _exprListAsUpdateFunction(self,args,exprList,wrapInputs,unwrapOutputs):
    pyfunReturningList = theano.function(inputs=args, outputs=exprList)
    # TOFIX make this take a list as the argument not X,Y
    def closure(X,Y):
      inputs = map(self._wrapMsg,[X,Y]) if wrapInputs else [X,Y]
      rawUpdates = pyfunReturningList(*inputs)
      if unwrapOutputs:
        result = map(lambda key,rawUpdate:(key,self._unwrapUpdate(key,rawUpdate)), self.ws.params, rawUpdates)
        return result
      else:
        return zip(self.ws.params, rawUpdates)
    return closure

  #
  # evaluators
  #

#  def eval(self,rawInputs):
#    inputs = map(self._wrapMsg,rawInputs)
#    return self._unwrapOutputs(self.ws.inferenceFun(*inputs))
#
#  def evalDataLoss(self,rawInputs,rawTarget):
#    # the loss depends on the rawInputs, which will usually be
#    # [x,target_y] and the parameters, which here are
#    # passed in as (pred,arity) keys
#    inputs = map(self._wrapMsg, rawInputs)
#    target = self._wrapMsg(rawTarget)
#    formalArgs = inputs + [target]
#    return self.unwrapOutput(self.ws.dataLossFun(*formalArgs))
#
#  def evalDataLossGrad(self,rawInputs,rawTarget):
#    # the loss depends on the rawInputs, which will usually be
#    # [x,target_y] and the parameters, which here are
#    # passed in as (pred,arity) keys
#    inputs = map(self._wrapMsg, rawInputs)
#    target = self._wrapMsg(rawTarget)
#    formalArgs = inputs + [target]
#    rawUpdates = self.ws.dataLossGradFun(*formalArgs)
#    return map(lambda key,rawUpdate:self._unwrapUpdate(key,rawUpdate), self.ws.params, rawUpdates)

  def _insertHandleExpr(self,key,name,val):
    self.ws._handleExpr[key] = self.ws._handleExprVar[key] = theano.shared(val, name=name)

  def _applyOpToNonzerosOfDense(self,op,expr):
    # useful subroutine
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)

  def show(self,verbose=0):
    """ print a summary of current workspace to stdout """
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

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TheanoCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def _createPlaceholder(self,name,kind,typeName):
    assert kind=='vector'
    result = TT.drow(name)
    return result

  def _wrapMsg(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat.todense()

  def _unwrapOutput(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  def _unwrapUpdate(self,key,up):
    return self._unwrapOutput(up)

  def _softmaxFun2Expr(self,subExpr,typeName):
    return self._applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self._nullSmoother[typeName])

  def _transposeMatrixExpr(self,mx):
    return mx.T

  def _vecMatMulExpr(self,v,m):
    return v.dot(m)

  def _componentwiseMulExpr(self,v1,v2):
    return v1*v2

  def _weightedVecExpr(self,vec,weighter):
    return vec * TT.sum(weighter, axis=1, keepdims=True)

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):

  def _wrapDBMatrix(self,mat):
    return mat

  def _vecMatMulExpr(self,v,m):
    return TSB.structured_dot(v,m)
