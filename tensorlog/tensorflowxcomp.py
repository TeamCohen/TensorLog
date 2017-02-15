import tensorflow as tf
import scipy.sparse as ss
import numpy as np
import tensorflow as tf
import logging

from tensorlog import funs
from tensorlog import ops
from tensorlog import xcomp
from tensorlog import expt

class TensorFlowCrossCompiler(xcomp.AbstractCrossCompiler):

  def __init__(self,db):
    # track things you need to initialize before evaluation. NOTE: we
    # need to set up tfVarsToInitialize before calling super.__init__
    # since super.__init__ creates variables.
    self.tfVarsToInitialize = []
    super(TensorFlowCrossCompiler,self).__init__(db)
    self.session = tf.Session()
    self.sessionInitialized = False

  #
  # tensorflow specific routines
  #

  def getSession(self):
    """ Return a session, which is the one used by default in 'eval'
    calls.
    """
    return self.session

  def ensureSessionInitialized(self):
    """ Make sure the varables in the session have been initialized,
    initializing them if needed
    """
    if not self.sessionInitialized:
      for var in self.tfVarsToInitialize:
        self.session.run(var.initializer)
      self.sessionInitialized = True

  def getInputName(self):
    """ String key for the input placeholder
    """
    assert len(self.ws.inferenceArgs)==1
    return self.ws.inferenceArgs[0].name

  def getTargetName(self):
    """ String key for the target-output placeholder
    """
    assert len(self.ws.dataLossArgs)==1
    return self.ws.dataLossArgs[0].name

  def getFeedDict(self,X,Y):
    return { self.getInputName(): X, self.getTargetName(): Y}

  def optimizeDataLoss(self,optimizer,rawX,rawY,epochs=1):
    X = self.wrapMsg(rawX)
    Y = self.wrapMsg(rawY)
    fd = self.getFeedDict(X,Y)
    step = optimizer.minimize(self.ws.dataLossExpr, var_list=self.ws.getParamVariables())
    self.ensureSessionInitialized()
    # check for uninitialized variables from the optimizer
    with self.session.as_default():
      self.session.run(tf.global_variables_initializer())
      for i in range(epochs):
        step.run(feed_dict=fd)

  def accuracy(self,rawX,rawY):
    X_val = self.wrapMsg(rawX)
    Y_val = self.wrapMsg(rawY)
    fd = self.getFeedDict(X_val,Y_val)
    Y_ = self.ws.inferenceExpr
    Y = self.ws.dataLossArgs[0]
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.ensureSessionInitialized()
    with self.session.as_default():
      return accuracy.eval(fd)

  def runExpt(self,prog=None,trainData=None,testData=None, targetMode=None,
              savedTestPredictions=None,savedTestExamples=None,savedTrainExamples=None,savedModel=None,
              optimizer=None, epochs=10):
    """ sort of similar to tensorlog.expt.Expt().run()
    """
    assert targetMode is not None,'targetMode must be specified'
    assert prog is not None,'prog must be specified'
    prog.setAllWeights()

    expt.Expt.timeAction('compiling and cross-compiling', lambda:self.compile(targetMode, prog.db.params))

    assert optimizer is None,'optimizers not supported yet'
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(self.ws.dataLossExpr, var_list=self.ws.getParamVariables())
    X = trainData.getX(targetMode)
    Y = trainData.getY(targetMode)
    TX = testData.getX(targetMode)
    TY = testData.getY(targetMode)

    def printLoss(msg,X,Y): print msg,self.evalDataLoss([X],Y)
    def printAccuracy(msg,X,Y): print msg,self.accuracy(X,Y)

    expt.Expt.timeAction('computing train loss',lambda:printLoss('initial train loss',X,Y))
    expt.Expt.timeAction('computing test loss',lambda:printLoss('initial test loss',X,Y))
    expt.Expt.timeAction('computing train accuracy',lambda:printAccuracy('initial train accuracy',X,Y))
    expt.Expt.timeAction('computing test accuracy',lambda:printAccuracy('initial test accuracy',X,Y))

    def train(X,Y):
      fd = self.getFeedDict(self.wrapMsg(X), self.wrapMsg(Y))
      self.ensureSessionInitialized()
      session = self.getSession()
      with session.as_default():
        for i in range(epochs):
          train_step.run(feed_dict=fd)

    expt.Expt.timeAction('training', lambda:train(X,Y))

    expt.Expt.timeAction('computing train loss',lambda:printLoss('final train loss',X,Y))
    expt.Expt.timeAction('computing test loss',lambda:printLoss('final test loss',X,Y))
    expt.Expt.timeAction('computing train accuracy',lambda:printAccuracy('final train accuracy',X,Y))
    expt.Expt.timeAction('computing test accuracy',lambda:printAccuracy('final test accuracy',X,Y))

    if savedModel:
      self.exportAllLearnedParams()
      expt.Expt.timeAction('saving trained model', lambda:prog.db.serialize(savedModel))

    def savePredictions(fileName):
      Y_ = self.eval([TX])
      expt.Expt.predictionAsProPPRSolutions(fileName,targetMode.functor,prog.db,TX,Y_)

    if savedTestPredictions:
      expt.Expt.timeAction('saving test predictions', lambda:savePredictions(savedTestPredictions))

    if savedTestExamples:
      expt.Expt.timeAction('saving test examples', lambda:testData.saveProPPRExamples(savedTestExamples,prog.db))

    if savedTrainExamples:
      expt.Expt.timeAction('saving train examples',lambda:trainData.saveProPPRExamples(savedTrainExamples,prog.db))

    if savedTestPredictions and savedTestExamples:
      print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' \
                % (savedTestExamples,savedTestPredictions)


  #
  # standard xcomp interface
  #

  def finalizeInference(self):
    pass

  def buildLossExpr(self,params):
    target_y = self.createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector')
    self.ws.dataLossArgs = [target_y]
    # we want to take the log of the non-zero entries and leave the
    # zero entries alone, so add 1 to all the zero indices, then take
    # a log of that.
    inferenceReplacing0With1 = tf.where(
        self.ws.inferenceExpr>0.0,
        self.ws.inferenceExpr,
        tf.ones(tf.shape(self.ws.inferenceExpr), tf.float64))
    self.ws.dataLossExpr = tf.reduce_sum(-target_y * tf.log(inferenceReplacing0With1))
    if params is not None:
      self.ws.params = params
      paramVars = map(lambda p:self.ws.getHandleExprVariable(p), params)
      self.ws.dataLossGradExprs = tf.gradients(self.ws.dataLossExpr,paramVars)

  def eval(self,rawInputs):
    inputs = map(self.wrapMsg,rawInputs)
    bindings = dict(zip(self.ws.inferenceArgs,inputs))
    return self.unwrapOutput(self._evalWithBindings(self.ws.inferenceExpr,bindings))

  def evalDataLoss(self,rawInputs,rawTarget):
    inputs = map(self.wrapMsg, rawInputs)
    target = self.wrapMsg(rawTarget)
    bindings = dict(zip(self.ws.inferenceArgs+self.ws.dataLossArgs,
                        inputs+[target]))
    return self.unwrapOutput(self._evalWithBindings(self.ws.dataLossExpr,bindings))

  def evalDataLossGrad(self,rawInputs,rawTarget):
    inputs = map(self.wrapMsg, rawInputs)
    target = self.wrapMsg(rawTarget)
    bindings = dict(zip(self.ws.inferenceArgs+self.ws.dataLossArgs,
                        inputs+[target]))
    rawUpdates = [self._evalWithBindings(expr,bindings)
                  for expr in self.ws.dataLossGradExprs]
    return map(lambda key,rawUpdate:self.unwrapUpdate(key,rawUpdate), self.ws.params, rawUpdates)

  def _evalWithBindings(self,expr,bindings):
    self.ensureSessionInitialized()
    with self.session.as_default():
      return expr.eval(feed_dict=bindings)

  def show(self,verbose=0):
    """ Print a summary of this workspace to stdout """
    print 'exprArgs',self.ws.inferenceArgs
    print 'expr',self.ws.inferenceExpr,'type',type(self.ws.inferenceExpr)
    if verbose>=1:
      TensorFlowCrossCompiler.pprintExpr(self.ws.inferenceExpr)

  @staticmethod
  def pprintExpr(expr,previouslySeen=None,depth=0,maxdepth=20):
    """ Print debug-level information on a tensorlog expression """
    if previouslySeen is None:
      previouslySeen=set()
    if depth>maxdepth:
      print '...'
    else:
      print '| '*(depth+1),
      op = expr.op
      print 'expr:',expr,'type','op',op.name,'optype',op.type
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


  def getLearnedParam(self,key):
    self.ensureSessionInitialized()
    with self.session.as_default():
      varVal = self.ws._handleExprVar[key].eval()
    # same logic works for param values as param updates
    return self.unwrapUpdate(key, varVal)

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TensorFlowCrossCompiler):

  def __init__(self,db):
    super(DenseMatDenseMsgCrossCompiler,self).__init__(db)
    self._denseMatIndices = [(i,j) for i in range(self.db.dim()) for j in range(self.db.dim())]
    self._denseVecIndices = [(0,i) for i in range(self.db.dim())]

  def createPlaceholder(self,name,kind):
    assert kind=='vector'
    result = tf.placeholder(tf.float64, shape=[None,self.db.dim()], name="tensorlog/"+name)
    return result

  def insertHandleExpr(self,key,name,val):
    v = tf.Variable(val, name="tensorlog/"+name)
    self.tfVarsToInitialize.append(v)
    self.ws._handleExpr[key] = self.ws._handleExprVar[key] = v

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

  def unwrapUpdate(self,key,up):
    return self.unwrapOutput(up)

  def unwrapOutput(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = ss.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  def softmaxFun2Expr(self,subExpr):
    # zeros are actually big numbers for the softmax,
    # so replace them with -20
    subExprReplacing0WithNeg20 = tf.where(
      subExpr>0.0,
      subExpr,
      tf.ones(tf.shape(subExpr), tf.float64)*(-10.0))
    return tf.nn.softmax(subExprReplacing0WithNeg20 + self.nullSmoothing)

  def transposeMatrixExpr(self,m):
    return tf.transpose(m)

  def vecMatMulExpr(self,v,m):
    return tf.matmul(v,m)

  def componentwiseMulExpr(self,v1,v2):
    return tf.multiply(v1,v2)

  def weightedVecExpr(self,vec,weighter):
    return tf.multiply(vec, tf.reduce_sum(weighter, axis=1, keep_dims=True))

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):

  def __init__(self,db):
    super(SparseMatDenseMsgCrossCompiler,self).__init__(db)
    # we will need to save the original indices/indptr representation
    # of each sparse matrix
    self.sparseMatInfo = {}

  def insertHandleExpr(self,key,name,val):
    (functor,arity) = key
    if arity<2:
      # vectors are dense so they are just stored as Variables
      v = tf.Variable(val, name="tensorlog/"+name)
      self.tfVarsToInitialize.append(v)
      self.ws._handleExpr[key] = self.ws._handleExprVar[key] = v
    else:
      # matrixes are sparse so we need to convert them into
      # a handle expression that stores a SparseTensor, and
      # do some additional bookkeeping.

      # first convert from scipy csr format of indices,indptr,data to
      # tensorflow's format, where the sparseindices are a 2-D tensor.
      sparseIndices = []
      n = self.db.dim()
      for i in range(n):
        for j in val.indices[val.indptr[i]:val.indptr[i+1]]:
          sparseIndices.append([i,j])
      # save the old shape and indices for the scipy matrix so we can
      # reconstruct a scipy matrix in unwrapUpdate.
      self.sparseMatInfo[key] = (val.indices,val.indptr,val.shape)
      # create the handle expression, and save a link back to the
      # underlying varable which will be optimized, ie., the 'values'
      # of the SparseTensor,
      indiceVar = tf.Variable(np.array(sparseIndices), name="tensorlog/%s_indices" % name)
      valueVar = tf.Variable(val.data, name="tensorlog/%s_values" % name)
      # TODO: the "valueVar+0.0" seems to be necessary to get a non-zero
      # gradient, but I don't understand why.  w/o this there is no "read"
      # node in for the variable in the graph and the gradient fails
      self.ws._handleExpr[key] = tf.SparseTensor(indiceVar,valueVar+0.0,[n,n])
      self.ws._handleExprVar[key] = valueVar
      # record the variables that need to be initialized
      self.tfVarsToInitialize.append(indiceVar)
      self.tfVarsToInitialize.append(valueVar)

  def unwrapUpdate(self,key,up):
    # we will optimize by updating the ws._handleExprVar's, which are,
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

  def wrapDBMatrix(self,mat):
    return mat

  def transposeMatrixExpr(self,m):
    return tf.sparse_transpose(m)

  def vecMatMulExpr(self,v,m):
    mT = tf.sparse_transpose(m)
    vT = tf.transpose(v)
    return tf.transpose(tf.sparse_tensor_dense_matmul(mT,vT))
