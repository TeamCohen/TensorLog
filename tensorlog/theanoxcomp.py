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

  def _buildLossExpr(self,mode):
    target_y = self._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self.ws.inferenceOutputType)
    self.ws.dataLossArgs = self.ws.inferenceArgs + [target_y]
    self.ws.dataLossExpr = (-target_y * self._applyOpToNonzerosOfDense(TT.log,self.ws.inferenceExpr)).mean()
    self.ws.dataLossGradExprs = theano.grad(self.ws.dataLossExpr, self.getParamVariables(mode))

  def _asOneInputFunction(self,arg1,expr,wrapInputs,unwrapOutputs):
    pyfun = theano.function(inputs=[arg1], outputs=expr)
    def closure(rawInput1):
       input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
       tmp = pyfun(input1)[0]
       return self._unwrapOutput(tmp) if unwrapOutputs else tmp
    return closure

  def _asTwoInputFunction(self,arg1,arg2,expr,wrapInputs,unwrapOutputs):
    pyfun = theano.function(inputs=[arg1,arg2], outputs=expr)
    def closure(rawInput1,rawInput2):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
      tmp = pyfun(input1,input2)[0]
      return self._unwrapOutput(tmp) if unwrapOutputs else tmp
    return closure

  def _exprListAsUpdateFunction(self,arg1,arg2,exprList,wrapInputs,unwrapOutputs):
    pyfunReturningList = theano.function(inputs=[arg1,arg2], outputs=exprList)
    def closure(rawInput1,rawInput2):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
      rawUpdates = pyfunReturningList(input1,input2)
      if unwrapOutputs:
        result = map(lambda key,rawUpdate:(key,self._unwrapUpdate(key,rawUpdate)), self.prog.getParamList(), rawUpdates)
        return result
      else:
        return zip(self.getParamList(), rawUpdates)
    return closure

  def _insertHandleExpr(self,key,name,val):
    self._handleExpr[key] = self._handleExprVar[key] = theano.shared(val, name=name)

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
