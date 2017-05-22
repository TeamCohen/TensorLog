import logging
import os
import numpy as np
import scipy.sparse as ss
import tensorflow as tf

from tensorlog import comline
from tensorlog import config
from tensorlog import funs
from tensorlog import ops
from tensorlog import xcomp
from tensorlog import expt
from tensorlog import util

class TensorFlowCrossCompiler(xcomp.AbstractCrossCompiler):

  def __init__(self,db,summaryFile=None):
    """If summaryFile is provided, save some extra information to pass on
    to tensorboard.
    """
    super(TensorFlowCrossCompiler,self).__init__(db)
    self.tfVarsToInitialize = []
    self.summaryFile = summaryFile
    self.session = None
    self.sessionInitialized = None
    logging.debug('TensorFlowCrossCompiler initialized %.3f Gb' % util.memusage())

  def close(self):
    if self.session is not None:
      self.session.close()

  #
  # tensorflow specific routines
  #

  # low-level training stuff

  def _ensureWrapped(self,X,Y,wrapped):
    return (X,Y) if wrapped else (self._wrapMsg(X),self._wrapMsg(Y))

  def _ensureUnwrapped(self,X,Y,wrapped):
    return (X,Y) if not wrapped else (self._unwrapOutput(X),self._unwrapOutput(Y))

  def setSession(self,session=None):
    """ Insert a session for the
    """
    self.session = session

  def ensureSessionInitialized(self):
    """ Make sure the varables in the session have been initialized,
    initializing them if needed
    """
    if self.session is None:
      logging.debug('creating session %.3f Gb' % util.memusage())
      self.session = tf.Session()
      logging.debug('session created %.3f Gb' % util.memusage())
    if not self.sessionInitialized:
      logging.debug('initializing session %.3f Gb' % util.memusage())
      for var in self.tfVarsToInitialize:
        self.session.run(var.initializer)
      self.sessionInitialized = True
      logging.debug('session initialized %.3f Gb' % util.memusage())

  def getInputName(self,mode,inputs=None):
    """ String key for the input placeholder
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    assert len(self._wsDict[mode].inferenceArgs)==1
    return self._wsDict[mode].inferenceArgs[0].name

  def getInputPlaceholder(self,mode,inputs=None):
    """ The input placeholder
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    assert len(self._wsDict[mode].inferenceArgs)==1
    return self._wsDict[mode].inferenceArgs[0]

  def getTargetOutputName(self,mode,inputs=None):
    """ String key for the target-output placeholder
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    assert len(self._wsDict[mode].dataLossArgs)==2
    return self._wsDict[mode].dataLossArgs[-1].name

  def getTargetOutputPlaceholder(self,mode,inputs=None):
    """  The target-output placeholder
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    assert len(self._wsDict[mode].dataLossArgs)==2
    return self._wsDict[mode].dataLossArgs[-1]

  def getFeedDict(self,mode,X,Y,wrapped=False,inputs=None):
    """ Create a feed dictionary for training based on X and Y
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    (X,Y) = self._ensureWrapped(X,Y,wrapped)
    return { self.getInputName(mode):X, self.getTargetOutputName(mode):Y }

  def optimizeDataLoss(self,mode,optimizer,X,Y,epochs=1,minibatchSize=0,wrapped=False):
    """ Train
    """
    self.ensureSessionInitialized()
    if self.summaryFile:
      self.summaryWriter = tf.summary.FileWriter(self.summaryFile, self.session.graph)
    def runAndSummarize(fd,i):
      if not self.summaryFile:
        self.session.run([trainStep],feed_dict=fd)
      else:
        (stepSummary, _) = self.session.run([self.summaryMergeAll,trainStep],feed_dict=fd)
        self.summaryWriter.add_summary(stepSummary,i)

    trainStep = optimizer.minimize(self.ws.dataLossExpr, var_list=self.getParamVariables(mode))
    self.ensureSessionInitialized()
    if not minibatchSize:
      fd = self.getFeedDict(mode,X,Y,wrapped)
      for i in range(epochs):
        runAndSummarize(fd,i)
    else:
      X1,Y1 = self._ensureUnwrapped(X,Y,wrapped)
      dset = dataset.Dataset({targetMode:X1},{targetMode:Y1})
      for i in range(epochs):
        for mode,miniX,miniY in dset.minibatchIterator(batchsize=minibatchSize):
          fd = self.getFeedDict(mode,miniX,miniY,wrapped=False)
          runAndSummarize(fd,i)

  def accuracy(self,mode,X,Y,wrapped=False,inputs=None):
    """ Return accuracy of a model on a test set
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    Xval,trueYVal = self._ensureWrapped(X,Y,wrapped)
    trueY = tf.placeholder(tf.float32, shape=trueYVal.shape, name="tensorlog/trueY")
    fd = { self.getInputName(mode):Xval, trueY.name:trueYVal }
    (_, Y_) = self.inference(mode)
    correct_prediction = tf.equal(tf.argmax(trueY,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.ensureSessionInitialized()
    with self.session.as_default():
      return accuracy.eval(fd)

  # higher-level training stuff

  def runExpt(self,prog=None,trainData=None,testData=None,targetMode=None,
              savedTestPredictions=None,savedTestExamples=None,savedTrainExamples=None,savedModel=None,
              optimizer=None, epochs=10, minibatchSize=0):
    """Similar to tensorlog.expt.Expt().run()
    """
    assert targetMode is not None,'targetMode must be specified'
    assert prog is not None,'prog must be specified'
    logging.debug('runExpt calling setAllWeights %.3f Gb' % util.memusage())
    prog.setAllWeights()
    logging.debug('runExpt finished setAllWeights %.3f Gb' % util.memusage())

    expt.Expt.timeAction('compiling and cross-compiling', lambda:self.ensureCompiled(targetMode,inputs=None))

    assert optimizer is None,'optimizers not supported yet'
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(self.ws.dataLossExpr, var_list=self.getParamVariables(targetMode))
    X = trainData.getX(targetMode)
    Y = trainData.getY(targetMode)
    X,Y = self._ensureWrapped(X,Y,False)
    TX = testData.getX(targetMode)
    TY = testData.getY(targetMode)
    TX,TY = self._ensureWrapped(TX,TY,False)

    lossFun = self.dataLossFunction(targetMode,wrapInputs=False,unwrapOutputs=False)
    def printLoss(msg,X,Y): print msg,lossFun(X,Y)
    def printAccuracy(msg,X,Y): print msg,self.accuracy(targetMode,X,Y,wrapped=True)

    expt.Expt.timeAction('computing train loss',lambda:printLoss('initial train loss',X,Y))
    expt.Expt.timeAction('computing test loss',lambda:printLoss('initial test loss',TX,TY))
    expt.Expt.timeAction('computing train accuracy',lambda:printAccuracy('initial train accuracy',X,Y))
    expt.Expt.timeAction('computing test accuracy',lambda:printAccuracy('initial test accuracy',TX,TY))

    expt.Expt.timeAction('training', lambda:self.optimizeDataLoss(targetMode,optimizer,X,Y,epochs=epochs,minibatchSize=minibatchSize,wrapped=True))

    expt.Expt.timeAction('computing train loss',lambda:printLoss('final train loss',X,Y))
    expt.Expt.timeAction('computing test loss',lambda:printLoss('final test loss',TX,TY))
    expt.Expt.timeAction('computing train accuracy',lambda:printAccuracy('final train accuracy',X,Y))
    expt.Expt.timeAction('computing test accuracy',lambda:printAccuracy('final test accuracy',TX,TY))

    if savedModel:
      self.exportAllLearnedParams()
      expt.Expt.timeAction('saving trained model', lambda:prog.db.serialize(savedModel))

    def savePredictions(fileName):
      inferenceFun = self.inferenceFunction(targetMode,wrapInputs=False,unwrapOutputs=True)
      Y_ = inferenceFun(TX)
      # Y_ is unwrapped, but need to get unwrapped version of TX from testData
      expt.Expt.predictionAsProPPRSolutions(fileName,targetMode.functor,prog.db,testData.getX(targetMode),Y_)

    if savedTestPredictions:
      expt.Expt.timeAction('saving test predictions', lambda:savePredictions(savedTestPredictions))
    if savedTestExamples:
      expt.Expt.timeAction('saving test examples', lambda:testData.saveProPPRExamples(savedTestExamples,prog.db))
    if savedTrainExamples:
      expt.Expt.timeAction('saving train examples',lambda:trainData.saveProPPRExamples(savedTrainExamples,prog.db))
    if savedTestPredictions and savedTestExamples:
      print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' % (savedTestExamples,savedTestPredictions)

  # debug stuff

  @staticmethod
  def pprintExpr(expr,previouslySeen=None,depth=0,maxdepth=20):
    """ Print debug-level information on a tensorlog expression """
    if previouslySeen is None:
      previouslySeen=set()
    if depth>maxdepth:
      print '...'
    else:
      tab = '| '*(depth+1),
      op = expr.op
      print '%sexpr:' % tab,expr,'type','op',op.name,'optype',op.type
    if not expr in previouslySeen:
      previouslySeen.add(expr)
      for inp in op.inputs:
        TensorFlowCrossCompiler.pprintExpr(inp,previouslySeen,depth=depth+1,maxdepth=maxdepth)

  @staticmethod
  def pprintAndLocateGradFailure(expr,vars,previouslySeen=None,depth=0,maxdepth=20):
    """ Print debug-level information on a tensorlog expression, and also
    give an indication of where a gradient computation failed.  """
    if previouslySeen is None:
      previouslySeen=set()
    def hasGrad(expr):
      try:
        return all(map(lambda g:g is not None, tf.gradients(expr,vars))),'ok'
      except Exception as ex:
        return False,ex
    if depth>maxdepth:
      print '...'
    else:
      op = expr.op
      stat,ex = hasGrad(expr)
      tab = '+ ' if stat else '| '
      print tab*(depth+1),expr,op.name,ex
      if not expr in previouslySeen:
        previouslySeen.add(expr)
        for inp in op.inputs:
          TensorFlowCrossCompiler.pprintAndLocateGradFailure(
            inp,vars,previouslySeen,depth=depth+1,maxdepth=maxdepth)

  #
  # standard xcomp interface
  #

  def _finalizeCompile(self,mode):
    #TODO does this work for multi-mode compilation?
    if self.summaryFile:
      self.summaryMergeAll = tf.summary.merge_all()

  def _buildLossExpr(self,mode):
    target_y = self._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self.ws.inferenceOutputType)
    self.ws.dataLossArgs = self.ws.inferenceArgs + [target_y]
    # we want to take the log of the non-zero entries and leave the
    # zero entries alone, so add 1 to all the zero indices, then take
    # a log of that.
    inferenceReplacing0With1 = tf.where(
        self.ws.inferenceExpr>0.0,
        self.ws.inferenceExpr,
        tf.ones(tf.shape(self.ws.inferenceExpr), tf.float32))
    self.ws.dataLossExpr = tf.reduce_sum(-target_y * tf.log(inferenceReplacing0With1))
    self.ws.dataLossGradExprs = tf.gradients(self.ws.dataLossExpr,self.getParamVariables(mode))

  def _asOneInputFunction(self,arg1,expr,wrapInputs,unwrapOutputs):
    def closure(rawInput1,session=None):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      bindings = {arg1:input1}
      return self._callAndUnwrap(expr,bindings,unwrapOutputs,session)
    return closure

  def _asTwoInputFunction(self,arg1,arg2,expr,wrapInputs,unwrapOutputs):
    def closure(rawInput1,rawInput2,session=None):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
      bindings = {arg1:input1,arg2:input2}
      return self._callAndUnwrap(expr,bindings,unwrapOutputs,session)
    return closure

  def _exprListAsUpdateFunction(self,arg1,arg2,exprList,wrapInputs,unwrapOutputs):
    def closure(rawInput1,rawInput2,session=None):
      input1 = self._wrapMsg(rawInput1) if wrapInputs else rawInput1
      input2 = self._wrapMsg(rawInput2) if wrapInputs else rawInput2
      bindings = {arg1:input1,arg2:input2}
      if session is None:
        self.ensureSessionInitialized()
        with self.session.as_default():
          rawUpdates = [expr.eval(feed_dict=bindings) for expr in exprList]
      else:
        with session.as_default():
          rawUpdates = [expr.eval(feed_dict=bindings) for expr in exprList]
      if unwrapOutputs:
        return map(lambda key,rawUpdate:(key,self._unwrapUpdate(key,rawUpdate)), self.prog.getParamList(), rawUpdates)
      else:
        return zip(self.prog.getParamList(), rawUpdates)
    return closure

  def _callAndUnwrap(self,expr,bindings,unwrapOutputs,session):
    # helper for _asXInputFunction's
    if session is None:
      self.ensureSessionInitialized()
      with self.session.as_default():
        tmp = expr.eval(feed_dict=bindings)
    else:
      with session.as_default():
        tmp = expr.eval(feed_dict=bindings)
    return self._unwrapOutput(tmp) if unwrapOutputs else tmp

  def show(self,verbose=0):
    """ Print a summary of current workspace to stdout """
    print 'exprArgs',self.ws.inferenceArgs
    print 'expr',self.ws.inferenceExpr,'type',type(self.ws.inferenceExpr)
    if verbose>=1:
      TensorFlowCrossCompiler.pprintExpr(self.ws.inferenceExpr)

  def getLearnedParam(self,key,session=None):
    if session is None:
      self.ensureSessionInitialized()
      with self.session.as_default():
        varVal = self._handleExpr[key].eval()
    else:
      with session.as_default():
        varVal = self._handleExpr[key].eval()
    # same logic works for param values as param updates
    functor,arity = key
    if arity==1:
      return self._unwrapDBVector(key,varVal)
    else:
      return self._unwrapDBMatrix(key,varVal)

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TensorFlowCrossCompiler):

  def __init__(self,db,summaryFile=None):
    super(DenseMatDenseMsgCrossCompiler,self).__init__(db,summaryFile=summaryFile)

  def _softPlusInverse(self,y):
    approxLogInf = 50.0
    # be careful, because you can't apply softplus inverse to zero
    # values, so first add 1 to all the zeros, by adding to all and
    # subtracting from the nonzeros.  nonzeros here must be the values
    # that are not zero, and don't underflow to zero when we add and
    # subtract
    nonzeroIndices = np.nonzero((y-approxLogInf) + approxLogInf)
    tmp = y + 1.0
    tmp[nonzeroIndices] -= 1.0
    result = np.log(np.exp(tmp)-1.0)
    # finally put something back in the zero positions which
    # is small enough that softplus brings it back to zero
    result -= approxLogInf
    result[nonzeroIndices] += approxLogInf
    return result

  def _createPlaceholder(self,name,kind,typeName):
    assert kind=='vector'
    result = tf.placeholder(tf.float32, shape=[None,self.db.dim(typeName)], name="tensorlog/"+name)
    return result

  def _insertHandleExpr(self,key,name,val):
    # parameters are passed through softplus so they don't get pushed to zero
    # we need to undo this transformation when you load them into variables...
    isTrainable = (key in self.db.paramSet)
    v = self._reparameterizeAndRecordVar(val,name,isTrainable)
    self._handleExprVar[key] = v
    self._handleExpr[key] = self._reparameterizedVarExpr(v,isTrainable)

  def _reparameterizeAndRecordVar(self,val,name,isTrainable):
    initVal = self._softPlusInverse(val) if (isTrainable and xcomp.conf.reparameterizeMatrices) else val
    v = tf.Variable(initVal, name="tensorlog/"+name, trainable=isTrainable)
    self.summarize(name,v)
    self.tfVarsToInitialize.append(v)
    return v

  def _reparameterizedVarExpr(self,var,isTrainable):
    return tf.nn.softplus(var) if (isTrainable and xcomp.conf.reparameterizeMatrices) else var

  def summarize(self,name,v):
    if self.summaryFile:
      tf.summary.scalar('summaries/%s/size' % name, tf.size(v))

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

  def _unwrapUpdate(self,key,up):
    return self._unwrapOutput(up)

  def _unwrapOutput(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = ss.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  def _unwrapDBVector(self,key,vec):
    return self._unwrapOutput(vec)

  def _unwrapDBMatrix(self,key,mat):
    return self._unwrapOutput(vec)

  def _softmaxFun2Expr(self,subExpr,typeName):
    # zeros are actually big numbers for the softmax,
    # so replace them with a big negative number
    subExprReplacing0WithNeg10 = tf.where(
        subExpr>0.0,
        subExpr,
        tf.ones(tf.shape(subExpr), tf.float32)*(-10.0))
    return tf.nn.softmax(subExprReplacing0WithNeg10 + self._nullSmoother[typeName])

  def _transposeMatrixExpr(self,m):
    return tf.transpose(m)

  def _vecMatMulExpr(self,v,m):
    return tf.matmul(v,m)

  def _componentwiseMulExpr(self,v1,v2):
    return tf.multiply(v1,v2)

  def _weightedVecExpr(self,vec,weighter):
    return tf.multiply(vec, tf.reduce_sum(weighter, axis=1, keep_dims=True))

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):

  def __init__(self,db,summaryFile=None):
    logging.debug('SparseMatDenseMsgCrossCompiler calling %r %.3f Gb' % (super(SparseMatDenseMsgCrossCompiler,self).__init__,util.memusage()))
    super(SparseMatDenseMsgCrossCompiler,self).__init__(db,summaryFile=summaryFile)
    logging.debug('SparseMatDenseMsgCrossCompiler finished super.__init__ %.3f Gb' % util.memusage())
    # we will need to save the original indices/indptr representation
    # of each sparse matrix
    self.sparseMatInfo = {}
    logging.debug('SparseMatDenseMsgCrossCompiler initialized %.3f Gb' % util.memusage())


  def _insertHandleExpr(self,key,name,val):
    (functor,arity) = key
    isTrainable = (key in self.db.paramSet)
    if arity<2:
      initVal = self._softPlusInverse(val) if isTrainable else val
      v = self._reparameterizeAndRecordVar(val,name,isTrainable)
      self._handleExprVar[key] = v
      self._handleExpr[key] = self._reparameterizedVarExpr(v,isTrainable)
    else:
      # matrixes are sparse so we need to convert them into
      # a handle expression that stores a SparseTensor, and
      # do some additional bookkeeping.

      # first convert from scipy csr format of indices,indptr,data to
      # tensorflow's format, where the sparseindices are a 2-D tensor.
      sparseIndices = []
      (nRows,nCols) = val.shape
      for i in range(nRows):
        for j in val.indices[val.indptr[i]:val.indptr[i+1]]:
          sparseIndices.append([i,j])
      logging.debug('%d sparseIndices for %d x %d relation %s: sparsity %g' %
                    (len(sparseIndices),nRows,nCols,functor,len(sparseIndices)/float(nRows*nCols)))
      # save the old shape and indices for the scipy matrix so we can
      # reconstruct a scipy matrix in unwrapUpdate.
      self.sparseMatInfo[key] = (val.indices,val.indptr,val.shape)
      # create the handle expression, and save a link back to the
      # underlying varable which will be optimized, ie., the 'values'
      # of the SparseTensor,
      indiceVar = tf.Variable(np.array(sparseIndices), name="tensorlog/%s_indices" % name)
      valueVar = self._reparameterizeAndRecordVar(val.data,name,isTrainable)
      # note: the "valueVar+0.0" seems to be necessary to get a non-zero
      # gradient, but I don't understand why.  w/o this there is no "read"
      # node in for the variable in the graph and the gradient fails
      self._handleExpr[key] = tf.SparseTensor(indiceVar,self._reparameterizedVarExpr(valueVar,isTrainable)+0.0,[nRows,nCols])
      self._handleExprVar[key] = valueVar
      # record the index variable, which also needs to be initialized
      self.tfVarsToInitialize.append(indiceVar)
      self.summarize("%s_indices" % name,indiceVar)

  def _unwrapDBVector(self,key,vec):
    return ss.csr_matrix(vec)

  def _unwrapDBMatrix(self,key,mat):
    (indices,indptr,shape) = self.sparseMatInfo[key]
    # note mat will be a SparseTensor in this case.....
    return ss.csr_matrix((mat.values,indices,indptr),shape=shape)


  def _unwrapUpdate(self,key,up):
    # we will optimize by updating the _handleExprVar's, which are,
    # for a SparseTensor, the value expressions.  to check gradients
    # and such we will need to convert the value updates to tensorlog
    # sparse matrix updates.
    (functor,arity) = key
    if arity==1:
      return ss.csr_matrix(up)
    elif arity==2:
      (indices,indptr,shape) = self.sparseMatInfo[key]
      return ss.csr_matrix((up,indices,indptr),shape=shape)
    else:
      assert False

  #
  # override the dense-matrix operations with sparse ones
  #

  def _wrapDBMatrix(self,mat):
    return mat

  def _transposeMatrixExpr(self,m):
    return tf.sparse_transpose(m)

  def _vecMatMulExpr(self,v,m):
    # TODO: the sparse_transpose below throws two op_scope deprecation warning
    mT = tf.sparse_transpose(m)
    vT = tf.transpose(v)
    result = tf.transpose(tf.sparse_tensor_dense_matmul(mT,vT))
    return result
