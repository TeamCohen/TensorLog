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

  def compile(self,fun,params=None):
    """Compile a tensorlog function to tensorflow.
    """
    (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
    self.inferenceFun = self.expr # call self.inferenceFun.evaluate() to evaluate?
    self._buildLossExpr(params)
    return self

  def _buildLossExpr(self,params):
    """ Add in the stuff relating to loss"""
    pass
#    target_y = self._vectorVar('_target_y')
#    self.dataTargetArgs = [target_y]
#    self.dataLossExpr = (target_y*self.applyOpToNonzerosOfDense(TT.log,self.expr)).mean()
#    self.dataLossFun = theano.function(
#        inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
#        outputs=(self.expr,self.dataLossExpr),
#        mode='DebugMode')
#    if params is not None:
#      self.params = params
#      paramVars = map(self.getSubExpr,params)
#      self.dataLossGradExprs = theano.grad(self.dataLossExpr, paramVars)
#      self.dataLossGradFun = theano.function(
#          inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
#          outputs=self.dataLossGradExprs,
#          mode='DebugMode')

  #
  # evaluators
  #

  def wrapSymbols(self,inputSyms):
    """ Convert a list of symbols to a list of one-hot vectors that can be sent to eval"""
    return map(lambda sym:self._wrapDBVector(self.db.onehot(sym)), inputSyms)

  def _unwrapOutputs(self,targetFunctionOutputs):
    return map(lambda v:self._sparsify(v), targetFunctionOutputs)

  def eval(self,inputs):
    # should call expr.eval(feed_dict=dict) - need placeholder for inputs
    #print 'now evaluating',inputs,self._secondaryArgBindings(),'==>',self.expr
    inputDict = {}
    assert len(inputs)==len(self.exprArgs)
    for i in range(len(inputs)):
      inputDict[self.exprArgs[i]] = inputs[i]
    for k,v in self.subexprCacheVarBindings.items():
      inputDict[k] = v
    with self.sess.as_default():
      return self._unwrapOutputs([self.expr.eval(feed_dict=inputDict)])

#    formalArgs = inputs +self._secondaryArgBindings()
#    return self._unwrapOutputs(self.inferenceFun(*formalArgs))
#
#  def evalDataLoss(self,rawInputs,rawTarget):
#    # the loss depends on the rawInputs, which will usually be
#    # [x,target_y] and the parameters, which here are
#    # passed in as (pred,arity) keys
#    inputs = map(self._wrapDBVector, rawInputs)
#    target = self._wrapDBVector(rawTarget)
#    formalArgs = inputs + [target] + self._secondaryArgBindings()
#    return self._unwrapOutputs(self.dataLossFun(*formalArgs))
#
#  def evalDataLossGrad(self,rawInputs,rawTarget):
#    # the loss depends on the rawInputs, which will usually be
#    # [x,target_y] and the parameters, which here are
#    # passed in as (pred,arity) keys
#    inputs = map(self._wrapDBVector, rawInputs)
#    target = self._wrapDBVector(rawTarget)
#    formalArgs = inputs + [target] + self._secondaryArgBindings()
#    return self._unwrapOutputs(self.dataLossGradFun(*formalArgs))
#

  def show(self):
    """ print a summary to stdout """
    print 'exprArgs',self.exprArgs
    print 'expr',self.expr,'type',type(self.expr)
    TensorFlowCrossCompiler.pprintExpr(self.expr)
    print 'subexpr cache:'
    for k,v in self.subexprCacheVarBindings.items():
      print ' |',k,'type',type(v)


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


###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################

class DenseMatDenseMsgCrossCompiler(TensorFlowCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def __init__(self,db):
    super(DenseMatDenseMsgCrossCompiler,self).__init__(db)
    self._denseMatIndices = [(i,j) for i in range(self.db.dim()) for j in range(self.db.dim())]
    self._denseVecIndices = [(0,i) for i in range(self.db.dim())]

  def _sparseFromDenseMat(self,expr):
    return tf.SparseTensor(self._denseMatIndices, tf.reshape(expr, [-1]), expr.get_shape())

  def _sparseFromDenseVec(self,expr):
    return tf.SparseTensor(self._denseVecIndices, tf.reshape(expr, [-1]), expr.get_shape())

  def _vectorVar(self,name):
    with tf.name_scope('tensorlog') as score:
      return tf.placeholder(tf.float32, shape=[1,self.db.dim()], name=name)

  def _matrixVar(self,name):
    with tf.name_scope('tensorlog') as score:
      return tf.placeholder(tf.float32, shape=[self.db.dim(),self.db.dim()], name=name)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat.todense()

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
      tfEnv = xcomp.NameSpacer(self.allocNamespace())
      seqInputs = []
      if sharedInputs==None:
        # create the list of theano variables, which should be
        # used as inputs to the expression
        for v in fun.opInputs:
          tfEnv[v] = self._vectorVar(tfEnv.internalName(v))
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

  # operator expressions for dense matrices
  def op2Expr(self,tfEnv,op,depth):
    """Extend the theano environment with an expression for the
    destination of the Operator, for dense matrices
    """
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose,lambda mx:tf.transpose(mx))
      return tf.matmul(tfEnv[op.src], mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode)
      return tf.matmul(self.ones(), tf.transpose(mExpr))
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
      return tf.multiply(tfEnv[op.vec], tf.reduce_sum(tfEnv[op.weighter], axis=1, keep_dims=True))
    else:
      assert False,'cannot cross-compile %r' % op
