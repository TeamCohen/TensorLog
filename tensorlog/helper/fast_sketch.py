from tensorlog.helper.countmin_embeddings import Sketcher,Sketcher2
import scipy
try:
  ss = scipy.sparse
except:
  import scipy.sparse as ss
import numpy as np
import collections

def global_unsketch(S, hashmat, t, n):
  # first, transform S into n rows according to each hashmat:
    s1 = S.multiply(hashmat)
    # next, drop any row that has != t cells set:
    #   first build a sparse matrix with the same shadow as s1,
    #   but whose values are all 1. 
    #   Then sum across rows to get the count of the set cells.
    #   Then create a boolean mask based on the count.
    rows,cols = s1.nonzero()
    filterdata = ss.csr_matrix( (np.ones(s1.nnz), (rows,cols)), shape = s1.shape).sum(axis=1) == t
    #   Get the row indices of the rows with t cells set
    rows,_ = filterdata.nonzero()
    #   Create a square filter to select just those rows from s1
    filter = ss.csr_matrix( (np.ones(len(rows)), (rows, rows)), shape = [s1.shape[0],s1.shape[0]])
    s1f = filter * s1 
    # Now each hot row of s1f has exactly t nonzeros i.e. the number of nz elements is a multiple of t
    
    # Reshape the nonzero elements into a ?xt block to take the row mins
    s2 = s1f.data.reshape([-1,t]).min(axis=1)
    # Finally build the final 1xn entity vector
    rows = np.zeros(len(s2),dtype='int')
    #   The column indices are the row indices of the hot rows of s1f
    cols = s1f.nonzero()[0].reshape((-1,t))[:,1]
    shape = (1,n)
    s3 = ss.csr_matrix( (s2, (rows,cols)), shape=shape)
    return s3

class FastSketcher(Sketcher):
  def __init__(self,db,k,delta):
    super(FastSketcher, self).__init__(db,k,delta)
  def unsketch(self,S):
    """ Approximate the matrix that would be sketched as S, i.e., an M so
    that M*hashmat = S.
    
    """
    return global_unsketch(S,self.hashmat, self.t, self.n)

class FastSketcher2(Sketcher2):
  def __init__(self,db,k,delta):
    super(FastSketcher2,self).__init__(db, k, delta)
    self.sketchmatArg1 = {}
    for rel in self.sketchmatsArg1:
      self.sketchmatArg1[rel] = sum(self.sketchmatsArg1[rel])
  def follow(self,rel,S):
    """ The analog of an operator X.dot(M) in sketch space.  Given S where
    X.dot(hashmat)=S, return the sketch for X.dot(M).
    """
    # nz_indices will be the indices of the coo_matrix for rel where
    # the row# hashes to something in S - unsketch these indices the
    # way we did before
    nz_indices  = global_unsketch(S,self.sketchmatArg1[rel],self.t,self.sketchmatWeights[rel].shape[1])
    #return nz_indices
    # multiply these indices by the corresponding weights, and then
    # move back to sketch space
    return self.sketchmatWeights[rel].multiply(nz_indices).dot(self.sketchmatArg2[rel])