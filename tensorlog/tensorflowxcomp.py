import tensorflow as tf
import scipy.sparse as ss
import numpy as np

from tensorlog import funs
from tensorlog import ops
from tensorlog import xcomp

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
  # tensorflow specific stuff
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
    """ String key for the input placeholder
    """
    assert len(self.ws.dataLossArgs)==1
    return self.ws.dataLossArgs[0].name

  def getFeedDict(self,X,Y):
    return { self.getInputName(): X, self.getTargetName(): Y}

  #
  # xcomp interface
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
    with tf.name_scope('tensorlog') as scope:
      result = tf.placeholder(tf.float64, shape=[None,self.db.dim()], name=name)
      return result

  def insertHandleExpr(self,key,name,val):
    with tf.name_scope('tensorlog') as scope:
      v = tf.Variable(val, name=name)
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
      with tf.name_scope('tensorlog') as scope:
        v = tf.Variable(val, name=name)
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
      with tf.name_scope('tensorlog') as scope:      
        indiceVar = tf.Variable(np.array(sparseIndices), name="%s_indices" % name)
        valueVar = tf.Variable(val.data, name="%s_values" % name)
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

