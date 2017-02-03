import tensorflow as tf
import scipy.sparse as ss
import numpy as np

from tensorlog import funs
from tensorlog import ops
from tensorlog import xcomp

class TensorFlowCrossCompiler(xcomp.AbstractCrossCompiler):

  def __init__(self,db):
    super(TensorFlowCrossCompiler,self).__init__(db)
    self.sess = tf.Session()

  def finalizeInference(self):
    pass

  def buildLossExpr(self,params):
    target_y = self.createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector')
    self.ws.dataLossArgs = [target_y]
    nonzeroIndices = tf.where(self.inferenceExpr > 0)
    self.ws.dataLossExpr = target_y * tf.SparseTensor(inferenceIndices, tf.log(inferenceValues), inferenceShape)
    if params is not None:
      self.ws.params = params
      paramVars = map(lambda p:self.ws[p], params)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      gradDict = dict(optimizer.compute_gradients(paramVars))
      self.ws.dataLossGradExprs = map(lambda p:gradDict[p], self.ws.sharedVariableList)

  def eval(self,rawInputs):
    bindings = dict(zip(self.ws.inferenceArgs,rawInputs))
    return _evalWithBindings(self.ws.inferenceExpr,bindings)

  def evalDataLoss(self,rawInputs,rawTarget):
    bindings = dict(zip(self.ws.inferenceArgs+self.ws.dataLossArgs,
                        rawInputs+[rawTarget]))
    return _evalWithBindings(self.ws.dataLossExpr,bindings)

  def evalDataGrad(self,rawInputs,rawTarget):
    bindings = dict(zip(self.ws.inferenceArgs+self.ws.dataLossArgs,
                        rawInputs+[rawTarget]))
    return [_evalWithBindings(expr,bindings) for expr in self.ws.dataLossGradExprs]

  def _evalWithBindings(self,expr,bindings):
    for param,var in self.ws.sharedVariable.items():
      self.sess.run(var.initializer)
    with self.sess.as_default():
      return self.unwrapOutputs([expr.eval(feed_dict=bindings)])

  def show(self,verbose=0):
    """ print a summary to stdout """
    print 'exprArgs',self.ws.inferenceArgs
    print 'expr',self.ws.inferenceExpr,'type',type(self.ws.inferenceExpr)
    if verbose>=1:
      TensorFlowCrossCompiler.pprintExpr(self.ws.inferenceExpr)

  @staticmethod
  def pprintExpr(expr,depth=0,maxdepth=20):
    if depth>maxdepth:
      print '...'
    else:
      print '| '*(depth+1),
      op = expr.op
      print 'expr:',expr,'type','op',op.name,'optype',op.type
      for inp in op.inputs:
        TensorFlowCrossCompiler.pprintExpr(inp,depth=depth+1,maxdepth=maxdepth)

  # helpers

  def _sparseFromDenseMat(self,expr):
    return tf.SparseTensor(self._denseMatIndices, tf.reshape(expr, [-1]), expr.get_shape())

  def _sparseFromDenseVec(self,expr):
    return tf.SparseTensor(self._denseVecIndices, tf.reshape(expr, [-1]), expr.get_shape())

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
      return tf.placeholder(tf.float64, shape=[1,self.db.dim()], name=name)

  def createSharedVar(self,name,val,kind):
    with tf.name_scope('tensorlog') as scope:
      return tf.Variable(val, name=name)

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
    sx = ss.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  def softmaxFun2Expr(self,subExpr):
    sparseResult = tf.sparse_softmax(self._sparseFromDenseVec(subExpr+self.nullSmoothing))
    return tf.sparse_tensor_to_dense(sparseResult)

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

class SparseMatDenseMsgCrossCompiler(TensorFlowCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def __init__(self,db):
    super(SparseMatDenseMsgCrossCompiler,self).__init__(db)
    self._denseMatIndices = [(i,j) for i in range(self.db.dim()) for j in range(self.db.dim())]
    self._denseVecIndices = [(0,i) for i in range(self.db.dim())]

  def _sparseFromDenseMat(self,expr):
    return tf.SparseTensor(self._denseMatIndices, tf.reshape(expr, [-1]), expr.get_shape())

  def _sparseFromDenseVec(self,expr):
    return tf.SparseTensor(self._denseVecIndices, tf.reshape(expr, [-1]), expr.get_shape())

  def createVectorVar(self,name):
    with tf.name_scope('tensorlog') as score:
      return tf.placeholder(tf.float32, shape=[1,self.db.dim()], name=name)

  def createMatrixVar(self,name):
    with tf.name_scope('tensorlog') as score:
      return tf.placeholder(tf.float32, shape=[self.db.dim(),self.db.dim()], name=name)

  def wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  # TODO: work on interface with matrix - somehow we need to
  # extend it or subclass it to handle the three parts
  # so that we can
  def wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a value used by the
    target language """
    coo_mat = mat.tocoo()
    indices = []
    for i in range(len(coo_mat.data)):
      indices.append((coo_mat.row[i],coo_mat.col[i]))
    return tf.SparseTensor(indices,coo_mat.data,coo_mat.shape)

  def _sparsify(self,x):
    """Convert a vector produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = ss.csr_matrix(x)
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
      # wrap inner function with softmax function - note tf handles
      # sparse softmax the way tensorlog does, ignoring zeros
      inputs,subExpr = self.fun2Expr(fun.fun,sharedInputs,depth=depth)
      sparseResult = tf.sparse_softmax(self._sparseFromDenseVec(subExpr+self.nullSmoothing))
      return (inputs, tf.sparse_tensor_to_dense(sparseResult))

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
      # tfEnv, a 'tensorflow environment', maps nameSpaced variables
      # from the OpSeqFunction's environment to the corresponding
      # theano subexpressions
      tfEnv = xcompself.allocNamespace()
      seqInputs = []
      if sharedInputs==None:
        # create the list of theano variables, which should be
        # used as inputs to the expression
        for v in fun.opInputs:
          tfEnv[v] = self.createVectorVar(tfEnv.internalName(v))
          seqInputs.append(tfEnv[v])
      else:
        # copy over the existing inputs to the new environment
        assert len(fun.opInputs)==len(sharedInputs)
        for i in range(len(fun.opInputs)):
          v = fun.opInputs[i]
          tfEnv[v] = sharedInputs[i]
          seqInputs.append(tfEnv[v])
      # fill in the theano environment appropriately
      for op in fun.ops:
        tfEnv[op.dst] = self.op2Expr(tfEnv,op,depth)
      # return the inputs and the expression for the
      # OpSeqFunction's output
      return (seqInputs, tfEnv[fun.ops[-1].dst])

    elif isinstance(fun,funs.NullFunction):
      return ([], self.zeros())

    else:
      assert False,'cannot cross-compile %r' % fun

  # operator expressions for sparse matrices
  def op2Expr(self,tfEnv,op,depth):
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose,lambda mx:tf.sparse_transpose(mx))
      return tf.matmul(tfEnv[op.src], mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode,False,lambda mx:tf.sparse_transpose(mx))
      # todo: fix the theano ones to do this also
      return tf.sparse_tensor_dense_matmul(mExpr,tf.transpose(self.ones()))
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return tf.multiply(tfEnv[op.src],tfEnv[op.src2])
    elif isinstance(op,ops.DefinedPredOp):
      _inputs,subExpr = self.fun2Expr(op.subfun,[tfEnv[op.src]],depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self.onehot(op.onehotConst)
    elif isinstance(op,ops.AssignVectorToVar):
      return self.vector(op.matMode)
    elif isinstance(op,ops.WeightedVec):
      return tf.multiply(tfEnv[op.vec], tf.sparse_reduce_sum(tfEnv[op.weighter], axis=1, keep_dims=True))
    else:
      assert False,'cannot cross-compile %r' % op
