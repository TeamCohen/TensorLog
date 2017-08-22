import tensorflow as tf

def go():
  dense = tf.Variable([[0,0,10,1,0,0],[0,0,-2,3,0,0]], dtype=tf.float32)
  sm1 = tf.nn.softmax(dense)

  denseReplacing0WithNeg10 = tf.where(
      dense>0.0,
      dense,
      tf.ones(tf.shape(dense), tf.float32)*(-10.0))
  sm2 = tf.nn.softmax(denseReplacing0WithNeg10)

  nz_indices = tf.where(tf.not_equal(dense, tf.constant(0, dtype=tf.float32)))
  nz_values = tf.gather_nd(dense,nz_indices)
  sparse = tf.SparseTensor(nz_indices, nz_values, dense.get_shape())
  sm3 = tf.sparse_softmax(sparse)
  dm3a = tf.sparse_to_dense(sm3.indices,sm3.get_shape(),sm3.values)
  dm3b = tf.scatter_nd(sm3.indices,sm3.values,dense.get_shape())

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  from tensorflow.python.framework import ops
  for v in nz_indices,nz_values,sparse,sm3,dm3a,dm3b:
    print 'gradient of op',v,ops.get_gradient_function(v.op)

  print 'dense sm - direct',session.run(sm1)
  print 'dense sm - with -10 trick',session.run(sm2)
  print 'sparse sm',session.run(sm3)
  print 'densified sparse sm - old',session.run(dm3a)
  print 'densified sparse sm - new',session.run(dm3a)

if __name__ == "__main__":
  print 'trying'
  go()
