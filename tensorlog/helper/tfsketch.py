from tensorlog.helper.countmin_embeddings import Sketcher as ScipySketcher, Sketcher2 as ScipySketcher2
from tensorlog import xcomp,mutil
import tensorflow as tf

EPSILON=1e-10
def global_tensor_unsketch(S, hashmat, t, n):
  s_shape = tf.shape(S)
  all_cols = tf.to_int32(tf.lin_space(0.0, tf.to_float(s_shape[-1]), s_shape[-1]))
  # first, transform S into n rows according to each hashmat:
  s1 = tf.multiply(S,hashmat)
  # next, drop any row that has != t cells set:
  #   first build a sparse matrix with the same shadow as s1,
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
  
  s3shape = tf.to_int64(tf.stack([s_shape[0],n]))
  s3 = tf.scatter_nd(tf.where(select),s2_mins,s3shape)
  
  return s3

class Sketcher(ScipySketcher):
  def __init__(self,xc,k,delta,verbose=True):
    self.xc = xc
    super(Sketcher,self).__init__(xc.db, k, delta, verbose)
  def init_hashmats(self):
    ScipySketcher.init_hashmats(self)
    self.tf_hashmat = self.xc._constantVector('sketch/hashmat',self.hashmat)
  def sketch(self, T):
    return tf.tensordot(T, self.tf_hashmat, 1)
  def unsketch(self, Ts):
    return global_tensor_unsketch(Ts, self.tf_hashmat, self.t, self.n)

class Sketcher2(ScipySketcher2,Sketcher):
  pass

if __name__ == '__main__':
  from tensorlog import simple,declare
  tlog = simple.Compiler(db="g16.db",prog="../../datasets/grid/grid.ppr")
  sk = Sketcher(tlog.xc,10,0.01)
  utype = 'THING'
  x1_expr = tlog.xc._onehot('1,1', utype)
  x2_expr = tlog.xc._onehot('1,2', utype)
  #x_expr = x1_expr
  x_expr = tf.stack([x1_expr,x2_expr])
  M_expr = tlog.xc._matrix(declare.asMode("edge/io"),utype)
  try:
    S_expr = sk.sketch(x_expr)
  except:
    raise
  x_approx_expr = global_tensor_unsketch(S_expr, sk.tf_hashmat, sk.t, sk.n)
  
  session=tf.Session()
  session.run(tf.global_variables_initializer())
  x = x_expr.eval(session=session)
  x_approx = session.run(x_approx_expr,feed_dict={})
  print str(x_approx)
    
    
    
    