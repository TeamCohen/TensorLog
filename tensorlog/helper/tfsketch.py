from tensorlog.helper.countmin_embeddings import Sketcher as ScipySketcher, Sketcher2 as ScipySketcher2
from tensorlog.helper.sketchadapters import SketchData as ScipySketchData
from tensorlog import tensorflowxcomp,ops,funs, dataset
import tensorflow as tf
from scipy.fftpack.realtransforms import dst

EPSILON=1e-10
def global_tensor_unsketch(S, hashmat, t, n):
  s_shape = tf.shape(S)
  with tf.control_dependencies([tf.assert_equal(tf.size(s_shape), 3, message="At unsketch time")]):
    # first, transform S into n rows according to each hashmat:
    s1 = tf.multiply(S,hashmat)
    # next, drop any row that has != t cells set:
    #   first build a matrix with the same shadow as s1,
    #   but whose values are all 1. 
    s1_nz_select = tf.square(s1)>EPSILON
    filterdata = tf.where(s1_nz_select,tf.ones_like(s1),s1)
    #   Then sum across rows to get the count of the set cells.
    #   Then create a dynamic partition based on the count.
    select = tf.equal(tf.reduce_sum(filterdata,axis=tf.size(s_shape)-1),t)
    filter = tf.where(select,tf.ones_like(select,dtype='int32'),tf.zeros_like(select,dtype='int32'))
    
    #   Partition to select just those rows from s1
    s1f = tf.dynamic_partition(s1, filter, 2)[1]
     
    # Reshape the nonzero elements into a ?xt block to take the row mins
    rowscols = tf.where(tf.square(s1f)>EPSILON)
    s2 = tf.reshape(tf.gather_nd(s1f,rowscols),shape=(-1,t))
    s2_mins = tf.reduce_min(s2,axis=1)
    
    # now drop the final values back into a batchsize x n zeros matrix
    s3shape = tf.to_int64(tf.stack([tf.to_int64(s_shape[0]),n]))
    s3 = tf.scatter_nd(tf.where(select),s2_mins,s3shape)
    
    return s3

def componentwise_tensor_min(X,Y):
  """ componentwise_min for two scipy matrices
  """
  data = tf.where(X<Y, X, Y)
  return data

class AbstractTfSketcher(ScipySketcher):
  def __init__(self,db,k,delta,verbose=True):
    super(AbstractTfSketcher,self).__init__(db, k, delta, verbose)
  def init_hashmats(self):
    ScipySketcher.init_hashmats(self)
  def init_xc(self,xc):
    self.xc = xc
    self.native={'hashmat':self.hashmat}
    self.hashmat = self.xc._constantVector('sketch/hashmat',self.hashmat)
    self.sparseMat = hasattr(self.xc, 'sparseMatInfo')
  def sketch(self, T):
    if T.shape.ndims == 2:
      #print "Expanding dims for tf sketch" 
      T=tf.expand_dims(T, axis=1)
    with tf.control_dependencies([tf.assert_equal(T.shape.ndims, 3, message="At sketch time")]):
      return tf.tensordot(T, self.hashmat, 1)
  def follow(self, mode, Ts, transpose=False):
    Mt = self.xc._matrix(mode,not transpose)
    MtSt = tf.sparse_tensor_dense_matmul(Mt,tf.transpose(self.unsketch(Ts)))
    return self.sketch(tf.transpose(MtSt))
  def unsketch(self, S):
    assert False, "Must override in subclass"

class Sketcher(AbstractTfSketcher):
  def __init__(self,db,k,delta,verbose=True):
    super(Sketcher,self).__init__(db, k, delta, verbose)
  def init_xc(self, xc):
    AbstractTfSketcher.init_xc(self, xc)
    def hashmatExpr(mat,i):
      return self.xc._constantVector('sketch/hashmat_%d'%i,mat)
    tmp = [hashmatExpr(m,i) for (i,m) in enumerate(self.hashmats)]
    self.hashmats = tmp
  def unsketch(self, S):
    result = tf.tensordot(S, tf.transpose(self.hashmats[0]),1)
    for d in range(1,self.t):
      xd = tf.tensordot(S, tf.transpose(self.hashmats[d]),1)
      result = componentwise_tensor_min(result,xd)
    return tf.squeeze(result,axis=1)

class FastSketcher(AbstractTfSketcher):
  def __init__(self,db,k,delta,verbose=True):
    super(FastSketcher,self).__init__(db, k, delta, verbose)
  def unsketch(self, Ts):
    return global_tensor_unsketch(Ts, self.hashmat, self.t, self.n)

def toTfKey(mode):
  return str(mode)

def scipyToTfExpr(xc,prefix,index):
  for key in index:
    index[key] = tf.to_float(xc._constantVector("sketch/"+prefix+toTfKey(key),index[key]))
    
def copyDict(src,dst):
  for key in src:
    dst[key] = src[key]
  return dst

class FastSketcher2(FastSketcher,ScipySketcher2):
  def __init__(self,db,k,delta,verbose=True):
    super(FastSketcher2,self).__init__(db, k, delta, verbose)
  def init_xc(self, xc):
    FastSketcher.init_xc(self, xc)
    
    self.sketchmatArg1 = {}
    for rel in self.sketchmatsArg1:
      self.sketchmatArg1[rel] = sum(self.sketchmatsArg1[rel])
    
    scipyToTfExpr(self.xc, "a1_", self.sketchmatArg1)
    scipyToTfExpr(self.xc,"a2_",self.sketchmatArg2)
    for matMode in self.sketchmatWeights:
      key = (matMode.functor,2)
      _ = self.xc._matrix(matMode) # so that xc does the necessary parameter bookkeeping
      self.sketchmatWeights[matMode] = self.xc._handleExpr[key].values
  def follow(self,mode,Ts,transpose=False):
    """ The analog of an operator X.dot(M) in sketch space.  Given S where
    X.dot(hashmat)=S, return the sketch for X.dot(M).
    """
    if transpose: # do we have a faster way to do this?
      mode = declare.asMode(str(mode)) # make a copy
      for i in 0,1: mode.prototype.args[i] = 'i' if mode.prototype.args[i]=='o' else 'o'
      
    # nz_indices will be the indices of the coo_matrix for mode where
    # the row# hashes to something in S - unsketch these indices the
    # way we did before
    follow_n = self.sketchmatWeights[mode].shape[-1]
    nz_indices  = global_tensor_unsketch(Ts,self.sketchmatArg1[mode],self.t,follow_n)
    #return nz_indices
    # multiply these indices by the corresponding weights, and then
    # move back to sketch space
    byweights = tf.multiply(self.sketchmatWeights[mode], nz_indices)
    ret = tf.tensordot(byweights,self.sketchmatArg2[mode],1)
    retshape = tf.expand_dims(ret,axis=1) 
    with tf.control_dependencies([tf.assert_equal(tf.size(tf.shape(retshape)), 3, message="After follow")]):
      return retshape

class SketchSMDMCrossCompiler(tensorflowxcomp.SparseMatDenseMsgCrossCompiler):
  def __init__(self,db,summaryFile=None):
    super(SketchSMDMCrossCompiler,self).__init__(db, summaryFile)
  def setSketcher(self,sk):
    self.sk = sk
    self.sk.init_xc(self)
  def _op2Expr(self, nspacer, op, depth):
    if isinstance(op,ops.VecMatMulOp):
      # self._vecMatMulExpr(nspacer[op.src], self._matrix(op.matMode,op.transpose))
      _=self._matrix(op.matMode,op.transpose)
      src = nspacer[op.src]
      if src.shape.ndims == 2:
        src = tf.expand_dims(src, 1)
      return self.sk.follow(op.matMode, src, op.transpose)
    elif isinstance(op,ops.AssignPreimageToVar):
      # self._vecMatMulExpr(self._ones(self._preimageOnesType(op.matMode)), self._matrix(op.matMode,True))
      return self.sk.sketch(self._vecMatMulExpr(self._ones(self._preimageOnesType(op.matMode)), self._matrix(op.matMode,True)))
    elif isinstance(op,ops.AssignOnehotToVar):
      # self._onehot(op.onehotConst,op.dstType)
      return self.sk.sketch(self._onehot(op.onehotConst,op.dstType))
    elif isinstance(op,ops.AssignVectorToVar):
      # self._vector(op.matMode)
      return self.sk.sketch(self._vector(op.matMode))
    elif isinstance(op,ops.WeightedVec):
      # self._weightedVecExpr(nspacer[op.vec], nspacer[op.weighter])
      return self._weightedVecExpr(nspacer[op.vec], nspacer[op.weighter])/self.sk.t
    else:
      return tensorflowxcomp.SparseMatDenseMsgCrossCompiler._op2Expr(self, nspacer, op, depth)
  def _fun2Expr(self, fun, sharedInputs=None, depth=0):
    if isinstance(fun,funs.NullFunction):
      typeName = self._wrapOutputType(fun)
      return ([], self.sk.sketch(self._zeros(typeName)), typeName)
    else:
      return tensorflowxcomp.SparseMatDenseMsgCrossCompiler._fun2Expr(self, fun, sharedInputs=sharedInputs, depth=depth)
  def _softmaxFun2Expr(self, subExpr, typeName):
    subExprReplacing0WithNeg10 = tf.where(
        subExpr>0.0,
        subExpr/self.sk.t,
        tf.ones(tf.shape(subExpr), tf.float32)*(-10.0))
    return tf.nn.softmax(subExprReplacing0WithNeg10 + self.sk.sketch(self._nullSmoother[typeName]))
  def _createPlaceholder(self,name,kind,typeName):
    assert kind=='vector'
    name=tensorflowxcomp.sanitizeVariableName(name)
    result = tf.placeholder(tf.float32, shape=[None,self.sk.m*self.sk.t], name="tensorlog/"+name)
    return result

def sketchDataset(sk, dset):
  """Convert an entity-space Dataset object to a SketchDataset (compatible with SketchLearner)"""
  xDict = {}
  yDict = {}
  for mode in dset.modesToLearn():
      try:
          x=dset.getX(mode)
          y=dset.getY(mode)
          xDict[mode] = x.dot(sk.native['hashmat'])
          yDict[mode] = y.dot(sk.native['hashmat'])
      except:
          print 'mode',mode
          raise
  return dataset.Dataset(xDict,yDict)

class SketchData(ScipySketchData):
  def __init__(self,sketcher,image,name="unnamed"):
    self.sketch = sketchDataset(sketcher,image)
    self.native = image
    self.name=name
    self.toggle('XY','sketch')

if __name__ == '__main__':
  from tensorlog import simple,declare
  tlog = simple.Compiler(db="g16.db",prog="../../datasets/grid/grid.ppr")
  sk = Sketcher(tlog.xc.db,10,0.01)
  sk.describe()
  sk.init_xc(tlog.xc)
  utype = 'THING'
  x1_expr = tlog.xc._onehot('1,1', utype)
  x2_expr = tlog.xc._onehot('1,2', utype)
  x_expr = x1_expr
  #x_expr = tf.stack([x1_expr,x2_expr])
  xs_expr = x_expr
  if xs_expr.shape.ndims >2: xs_expr = tf.squeeze(x_expr)
  mode = declare.asMode("edge/io")
  M_expr = tlog.xc._matrix(mode,utype)
  S_expr = sk.sketch(x_expr)
  x_approx_expr = sk.unsketch(S_expr)
  
  print "xs_expr",xs_expr
  print "M_expr",M_expr
  xn_expr = tlog.xc._vecMatMulExpr(xs_expr, M_expr)
  
  Sn_expr = sk.follow(mode,S_expr)
  xn_approx_expr = sk.unsketch(Sn_expr)
  
  session=tf.Session()
  session.run(tf.global_variables_initializer())
  x = xs_expr.eval(session=session)
  x_approx = session.run(x_approx_expr,feed_dict={})
  print "x",x.shape
  print str(x)
  print "x_approx",x_approx.shape
  print str(x_approx)
  xn = xn_expr.eval(session=session)
  Sn = Sn_expr.eval(session=session)
  xn_approx = session.run(xn_approx_expr,feed_dict={})
  print "xn_approx"
  print str(xn_approx)
  print "xn"
  print str(xn)
#   print "error"
#   print str(xn_approx - xn)
    
    
    
    
