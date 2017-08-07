# random test/experimental code that william unaccountably wanted to
# check into git

import math
import numpy as np
import scipy.sparse as sp

import tensorflow as tf

from tensorlog import matrixdb,mutil

def go():
  db = matrixdb.MatrixDB.loadFile("test-data/textcattoy_corpus.cfacts")
  m = db.matEncoding[('hasWord',2)]
  print 'm',m.shape,m.nnz
  def tfidf_transform(tf_matrix):
    # implements Salton's TFIDF transformation, ie l2-normalized
    # vector after scaling by: log(tf+1.0) * log(#docs/idf)
    df = tf_matrix.sum(axis=0)
    # count docs by summing word counts w/in each row of m,
    # clipping the sum down to 1, and and then adding up
    ndoc = np.clip(tf_matrix.sum(axis=1),0.0,1.0).sum()
    # this ensures idfs will be zero for any terms with df==0, after I
    # take the log
    df[df==0] = ndoc
    idf = np.log( np.reciprocal(df) * ndoc )
    # first compute log(tf+1.0)
    scaled_tf_matrix = mutil.mapData(lambda v:np.log(v+1.0),tf_matrix)
    # now multiply by idf factor
    unnormalized_tfidf_matrix = mutil.broadcastAndComponentwiseMultiply(scaled_tf_matrix, sp.csr_matrix(idf))
    # compute normalizer needed for l2 normalization
    normalizer = mutil.mapData(lambda v:v*v, unnormalized_tfidf_matrix).sum(axis=1)
    normalizer = np.sqrt(normalizer)
    # finally, multiply unnormalized_tfidf_matrix and normalizer,
    # which is complicated since they are sparse, scipy.sparse doesn't
    # support broadcasting, and
    # mutil.broadcastAndComponentwiseMultiply can only broadcast along
    # rows. so we need to transpose everything before and after, and
    # convert into csr matrices
    tmp1 = sp.csr_matrix(unnormalized_tfidf_matrix.transpose())
    tmp2 = sp.csr_matrix(normalizer.transpose())
    tmp2.eliminate_zeros()
    tmp3 = mutil.mapData(np.reciprocal,tmp2)
    tmp4 = mutil.broadcastAndComponentwiseMultiply(tmp1,tmp3)
    result = sp.csr_matrix(tmp4.transpose())
    return result
  m = tfidf_transform(m)

def go1():
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
