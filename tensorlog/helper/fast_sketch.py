from tensorlog.helper.countmin_embeddings import Sketcher,Sketcher2
from tensorlog import declare
import scipy
try:
  ss = scipy.sparse
except:
  import scipy.sparse as ss
import numpy as np
import collections

def global_unsketch(S, hashmat, t, n):
  # first, transform S into n rows according to each hashmat:
  #print "S",S.shape
  #print "hashmat",hashmat.shape
  
  s3_data = []
  s3_rows = []
  s3_cols = []
  k=0
  for ri in range(S.shape[0]):
    si = S.getrow(ri)
    #print "si",si.shape
    s1 =  si.multiply(hashmat)
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
    
    if s1f.nnz == 0: continue
    assert s1f.nnz >= t, "bad s1f nnz: %d vs t=%d" %(s1f.nnz,t)
    # Reshape the nonzero elements into a ?xt block to take the row mins
    s2 = s1f.data.reshape([-1,t]).min(axis=1)
    # Finally build the final 1xn entity vector
    rows = ri*np.ones(len(s2),dtype='int')
    #   The column indices are the row indices of the hot rows of s1f
    cols = s1f.nonzero()[0].reshape((-1,t))[:,1]
    try:
      s3_data.extend(s2)
      s3_rows.extend(rows)
      s3_cols.extend(cols)
      k += len(s2)
    except ValueError:
      print "S",S.shape
      print "si",si.shape
      print "s2",s2.shape
      print "[buffer]", s3_data
      print "rows",rows.shape
      print "[buffer]", s3_rows
      print "cols",cols.shape
      print "[buffer]", s3_cols
      print "s1f.nnz", s1f.nnz
      raise
  shape = (S.shape[0],n)
  s3 = ss.csr_matrix( (s3_data, (s3_rows,s3_cols)), shape=shape)
  assert s3.nnz == k, "bad combo: nnz = %d when k = %d" % (s3.nnz,k)
  return s3

class FastSketcher(Sketcher):
  def __init__(self,db,k,delta,verbose=True):
    super(FastSketcher, self).__init__(db,k,delta,verbose)
  def unsketch(self,S):
    """ Approximate the matrix that would be sketched as S, i.e., an M so
    that M*hashmat = S.
    
    """
    return global_unsketch(S,self.hashmat, self.t, self.n)

class FastSketcher2(Sketcher2):
  def __init__(self,db,k,delta,verbose=True):
    super(FastSketcher2,self).__init__(db, k, delta,verbose)
    self.sketchmatArg1 = {}
    for rel in self.sketchmatsArg1:
      self.sketchmatArg1[rel] = sum(self.sketchmatsArg1[rel])
  def follow(self,mode,S,transpose=False):
    """ The analog of an operator X.dot(M) in sketch space.  Given S where
    X.dot(hashmat)=S, return the sketch for X.dot(M).
    """
    if transpose: # do we have a faster way to do this?
      mode = declare.asMode(str(mode)) # make a copy
      for i in 0,1: mode.prototype.args[i] = 'i' if mode.prototype.args[i]=='o' else 'o'
      
    # nz_indices will be the indices of the coo_matrix for mode where
    # the row# hashes to something in S - unsketch these indices the
    # way we did before
      
    nz_indices  = global_unsketch(S,self.sketchmatArg1[mode],self.t,self.sketchmatWeights[mode].shape[1])
    #return nz_indices
    # multiply these indices by the corresponding weights, and then
    # move back to sketch space
    return self.sketchmatWeights[mode].multiply(nz_indices).dot(self.sketchmatArg2[mode])
  def unsketch(self,S):
    """ Approximate the matrix that would be sketched as S, i.e., an M so
    that M*hashmat = S.
    
    """
    return global_unsketch(S,self.hashmat, self.t, self.n)
  
  