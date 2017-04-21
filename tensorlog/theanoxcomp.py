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
from tensorlog import dataset
from tensorlog import xcomp
from tensorlog import learnxcomp
#from tensorlog.debug import mode

class TheanoCrossCompiler(xcomp.AbstractCrossCompiler):

  def _buildLossExpr(self,mode):
    target_y = self._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self._wsDict[mode].inferenceOutputType)
    self._wsDict[mode].dataLossArgs = self._wsDict[mode].inferenceArgs + [target_y]
    placeholder = map(lambda x:0*x, self.getParamVariables(mode)) # theano doesn't like it when some paramVariables don't appear in the loss expr
    self._wsDict[mode].dataLossExpr = (-target_y * self._applyOpToNonzerosOfDense(TT.log,self._wsDict[mode].inferenceExpr)+sum(placeholder)).mean()
    self._wsDict[mode].dataLossGradExprs = theano.grad(self._wsDict[mode].dataLossExpr, self.getParamVariables(mode))

  def _asOneInputFunction(self,arg1,expr,wrapInputs,unwrapOutputs):
    pyfun = theano.function(inputs=[arg1], outputs=expr)
    def closure(rawInput1):
       input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
       tmp = pyfun(input1) # was [0] here -- not sure why. -kmm
       return self._unwrapOutput(tmp) if unwrapOutputs else tmp
    return closure

  def _asTwoInputFunction(self,arg1,arg2,expr,wrapInputs,unwrapOutputs):
    pyfun = theano.function(inputs=[arg1,arg2], outputs=expr)
#     print "arg1",arg1
#     print "arg2",arg2
    def closure(rawInput1,rawInput2):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
      tmp = pyfun(input1,input2) # was [0] here -- not sure why. -kmm
      return self._unwrapOutput(tmp) if unwrapOutputs else tmp
    return closure

  def _exprListAsUpdateFunction(self,arg1,arg2,exprList,wrapInputs,unwrapOutputs):
    pyfunReturningList = theano.function(inputs=[arg1,arg2], outputs=exprList, )
#     print "arg1",arg1
#     print "arg2",arg2
    def closure(rawInput1,rawInput2):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
#       print "arg1",rawInput1.shape
#       print "arg2",rawInput2.shape
      #print theano.printing.debugprint(pyfunReturningList)
      rawUpdates = pyfunReturningList(input1,input2)
      if unwrapOutputs:
        result = map(lambda key,rawUpdate:(key,self._unwrapUpdate(key,rawUpdate)), self.prog.getParamList(), rawUpdates)
        return result
      else:
        return zip(self.getParamList(), rawUpdates)
    return closure

  def _insertHandleExpr(self,key,name,val,broadcast=False):
    kwargs={}
    if broadcast: kwargs['broadcastable']=tuple([dim==1 for dim in val.shape])
    self._handleExpr[key] = self._handleExprVar[key] = theano.shared(val, name=name, **kwargs)
    #print "handleExpr %s shape"%name,val.shape,"broadcastable",self._handleExprVar[key].broadcastable
    

  def _applyOpToNonzerosOfDense(self,op,expr):
    # useful subroutine
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)
  
  def optimizeDataLoss(self,mode,optimizer,X,Y,epochs=1,minibatchSize=0,wrapped=False):
    mode = self.ensureCompiled(mode)
    try:
      has = mode in self._trainStepDict
    except:
      self._trainStepDict = {}
      has=False
    if has:
      trainStep = self._trainStepDict[mode]
    else:
      trainStep = self._trainStepDict[mode] = optimizer.minimize(self._wsDict[mode].dataLossExpr, var_list=self.getParamVariables(mode), 
                                 inputs=[self._wsDict[mode].inferenceArgs[0], self._wsDict[mode].dataLossArgs[-1]])
    if not minibatchSize:
      (X,Y) = self._ensureWrapped(X,Y,wrapped)
      for i in range(epochs):
        loss = trainStep(X,Y)
    else:
      X1,Y1 = self._ensureUnwrapped(X,Y,wrapped)
      dset = dataset.Dataset({mode:X1},{mode:Y1})
      for i in range(epochs):
        for mode,miniX,miniY in dset.minibatchIterator(batchsize=minibatchSize):
          (miniX,miniY) = self._ensureWrapped(miniX,miniY,wrapped)
          loss = trainStep(X,Y)
  
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
        
  def getLearnedParam(self,key,session=None):
    varVal = self._handleExprVar[key].eval()
    # same logic works for param values as param updates
    return self._unwrapUpdate(key, varVal)

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TheanoCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def _createPlaceholder(self,name,kind,typeName):
    assert kind=='vector'
    result = TT.dmatrix(name)
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

  def _unwrapDBVector(self,key,vec):
    return self._unwrapOutput(vec)

  def _unwrapDBMatrix(self,key,mat):
    return self._unwrapOutput(mat)

  def _softmaxFun2Expr(self,subExpr,typeName):
    # _applyTopToNonzerosOfDense overweights the null element by at least 0.05,
    # more than our desired margin of error. Fussing with the null smoothing didn't
    # help. 
    # tf doesn't have this problem -- it uses a -20 mask on the zero values.
    # in theano this would be something like -20*TT.isclose(subExpr,TT.zeros_like(subExpr))
    # but TT.isclose() is slow, so we use TT.exp(-s^2) as a faster approximation and 
    # cross our fingers and toes we don't have anything important in 0<s<1
    return TNN.nnet.softmax(subExpr+self._nullSmoother[typeName]-20*TT.exp(-subExpr*subExpr))
    #return self._applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self._nullSmoother[typeName])

  def _transposeMatrixExpr(self,mx):
    return mx.T

  def _vecMatMulExpr(self,v,m):
#     if not hasattr(self,"doti"): self.doti=0
#     self.doti+=1
#     v_printed=theano.printing.Print("v%d:"%self.doti,["shape"])(v)
#     m_printed=theano.printing.Print("m%d:"%self.doti,["shape"])(m)
#     return TT.dot(v_printed,m_printed) #v.dot(m)
    return TT.dot(v,m)

  def _componentwiseMulExpr(self,v1,v2):
#     if not hasattr(self,"cwi"): self.cwi=0
#     self.cwi+=1
#     print "v1.%d broadcastable "%self.cwi,v1.broadcastable
#     print "v2.%d broadcastable "%self.cwi,v2.broadcastable
#     v1_printed=theano.printing.Print("v1.%d"%self.cwi,["shape"])(v1)
#     v2_printed=theano.printing.Print("v2.%d"%self.cwi,["shape"])(v2)
#     return v1_printed*v2_printed
    return v1 * v2

  def _weightedVecExpr(self,vec,weighter):
#     if not hasattr(self,"wvi"): self.wvi=0
#     self.wvi+=1
#     vec_printed=theano.printing.Print("vec%d"%self.wvi,["shape"])(vec)
#     weighter_printed=theano.printing.Print("weighter%d"%self.wvi,["shape"])(weighter)
#     return vec_printed * TT.sum(weighter_printed, axis=1, keepdims=True)
    return vec * TT.sum(weighter, axis=1, keepdims=True)

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):

  def _wrapDBMatrix(self,mat):
    return mat

  def _vecMatMulExpr(self,v,m):
    return TSB.structured_dot(v,m)

###############################################################################
# learning
###############################################################################

class Optimizer(object):
  def __init__(self):
    pass
  def minimize(self,expr,var_list=[]):
    """Return a training step for optimizing expr with respect to var_list.
    """
    assert False,'abstract method called'

class GD(Optimizer):
  def __init__(self,learning_rate):
    super(GD,self).__init__()
    self.learning_rate = learning_rate
  def minimize(self, expr, var_list=[], inputs=[]):
    dlosses = TT.grad(expr, var_list)
    updates = [(v, v 
                - TT.cast(self.learning_rate,v.dtype) 
                * (TT.cast(dloss,v.dtype) if isinstance(dloss.type,TT.type.TensorType) else dloss)) 
               for v,dloss in zip(var_list,dlosses)]
    trainStep = theano.function(inputs, expr, updates=updates, )
    return trainStep


class FixedRateGDLearner(learnxcomp.BatchEpochsLearner):
    """ A gradient descent learner.
    """

    def __init__(self,prog,xc=None,compilerClass=DenseMatDenseMsgCrossCompiler,epochs=20,rate=0.1,regularizer=None,tracer=None,epochTracer=None):
        super(FixedRateGDLearner,self).__init__(prog,xc,epochs=epochs,compilerClass=compilerClass,regularizer=regularizer,tracer=tracer,epochTracer=epochTracer)
        self.rate=rate
        self.optimizer = GD(learning_rate=rate)
    
    def trainMode(self,mode,X,Y,epochs=-1):
      if epochs<0: epochs=self.epochs
      self.xc.optimizeDataLoss(mode,self.optimizer,X,Y,epochs=epochs)
