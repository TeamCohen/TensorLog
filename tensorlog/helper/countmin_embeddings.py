# first draft at code for doing count-min embeddings

import numpy as np
import scipy
import math
import random
import sys
import getopt

from tensorlog import matrixdb,comline

LOGBASE = 2
MAX_NUM_HASHES = 100

def componentwise_min(X,Y):
  """ my old version scipy doesn't have csr.minimum(other) implemented!
  """
  X1 = scipy.sparse.csr_matrix(X)
  X1.data = np.ones_like(X.data)
  Y1 = scipy.sparse.csr_matrix(Y)
  Y1.data = np.ones_like(Y.data)
  commonIndicesXVals = X.multiply(Y1)
  commonIndicesYVals = Y.multiply(X1)
  data = np.minimum(commonIndicesXVals.data,commonIndicesYVals.data)
  result = scipy.sparse.csr_matrix((data,commonIndicesXVals.indices,commonIndicesXVals.indptr),shape=X1.shape,dtype='float32')
  return result

class Sketcher(object):

  def __init__(self,db,k,delta):
    """ Follows notation in my notes.
    k = max set size
    delta = prob of any error in unsketch
    t,m = hash functions map to one of t subranges each of width [1...m]
    n = original dimension

    I usually index 0 ... t with d for some reason
    """

    self.db = db
    self.n = db.dim()
    self.k = k
    self.delta = delta
    self.m = int(math.ceil(LOGBASE * k))
    self.t = int(math.ceil(-math.log(delta,LOGBASE)))
    self.hash_salt = [ random.getrandbits(32) for _ in range(MAX_NUM_HASHES) ]
    self.offset = hash("389437892342389")
    self.init_hashmats()
    print 'n',self.n,'t',self.t,'m',self.m,'sketch size',self.t*self.m,'compression',float(self.n)/(self.t*self.m)

  def init_hashmats(self):
    """ self.hashmat caches out sketch for each object 0...n
    self.hashmats[d] is the part that hashes to the d'th subrange.
    """
    # self.hashmats[d] is a matrix version of hash_function(d,x)
    self.hashmats = []
    # self.hashmat is sum of self.hashmats - init with an empty csr matrix
    coo_mat = scipy.sparse.coo_matrix(([],([],[])), shape=(self.n,self.m*self.t))
    self.hashmat = scipy.sparse.csr_matrix(coo_mat,dtype='float32')
      
    # cache out each hash function
    databuf = np.ones(self.n)
    for d in range(self.t):
      rowbuf = []
      colbuf = []
      for i in range(self.n):
        j = self.hash_function(d,i)
        rowbuf.append(i)
        colbuf.append(j)
      coo_mat = scipy.sparse.coo_matrix((databuf,(rowbuf,colbuf)), shape=(self.n,self.m*self.t))
      hd = scipy.sparse.csr_matrix(coo_mat,dtype='float32')
      self.hashmats.append(hd)
      self.hashmat = self.hashmat + hd
      #print 'hashmat',d,type(self.hashmats[d]),self.hashmats[d].shape
    #print 'hashmat',type(self.hashmat),self.hashmat.shape

  def hash_function(self,d,x): 
    """ The d-th hash function, which maps x into the range [m*d, ..., m*(d+1)]
    """
    return ((hash(x + self.offset)^self.hash_salt[d]) % self.m) + self.m*d

  def sketch(self,X):
    """ produce a sketch for X
    """
    return X.dot(self.hashmat)

  def unsketch(self,S):
    """ Approximate the matrix that would be sketched as S, i.e., an M so
    that M*hashmat = S """
    #print 'unsketching',S.shape,'with h0 shape', self.hashmats[0].shape,self.hashmats[0].transpose().shape
    result = scipy.sparse.csr_matrix(S.dot(self.hashmats[0].transpose()))
    #print 'result',type(result),result.shape
    for d in range(1,self.t):
      xd = scipy.sparse.csr_matrix(S.dot(self.hashmats[d].transpose()))
      result = componentwise_min(result,xd)
    return result

  def follow(self,rel,S):
    """ The analog of x.dot(M) in sketch space
    """
    M = self.db.matEncoding[(rel,2)]
    return self.unsketch(S).dot(M).dot(self.hashmat)

  def showme(self,**kw): 
    """ For debugging - print a summary of some objects, especially
    usedful for checking sparse matrixes
    """
    for tag,val in kw.items():
      try:
        rows,cols = val.shape
        print tag,'=',rows,'x',cols,val.__class__.__name__,
        nz = val.nnz
        print nz,'nonzeros',
        if cols==self.db.dim() and nz<10:
          if rows>1:
            print self.db.matrixAsSymbolDict(val),
          else:
            print self.db.rowAsSymbolDict(val),        
      except AttributeError:
        print tag,'=',val
      print

  def compareRows(self,gold,approx):
    dGold = self.db.rowAsSymbolDict(gold)
    dApprox = self.db.rowAsSymbolDict(approx)
    fp = set(dApprox.keys()) - set(dGold.keys())
    fn = set(dGold.keys()) - set(dApprox.keys())
    print 'errors:',len(fp),list(fp)[0:10]
    if len(fn)>0:
      print '  also there are false negatives - really?',fn

# example: python countmin_embeddings.py --db  'fb15k.db|../../datasets/fb15k-speed/inputs/fb15k-valid.cfacts'  --x m_x_02md_2 --rel american_football_x_football_coach_position_x_coaches_holding_this_position_x_american_football_x_football_historical_coach_position_x_team --k 50
# example: python countmin_embeddings.py --db  'g64.db|../../datasets/grid/inputs/g64.cfacts'  --x 25,36 --rel edge --k 10

if __name__ == "__main__":
  print 'loading db...'
  optlist,args = getopt.getopt(sys.argv[1:],'x',["db=","x=", "rel=","k=","delta="])
  optdict = dict(optlist)
  #db = matrixdb.MatrixDB.loadFile('g10.cfacts')
  print 'optdict',optdict
  db = comline.parseDBSpec(optdict.get('--db','g10.cfacts'))
  xsym = optdict.get('--x','5,5')
  rel = optdict.get('--rel','edge')
  k = int(optdict.get('--k','10'))
  delta = float(optdict.get('--delta','0.01'))

  M_edge = db.matEncoding[(rel,2)]
  M_edge.data[M_edge.data>0] = 1.0

  sk = Sketcher(db,k,delta/db.dim())

  def probe(x):
    # find a sketch for x and then invert it
    print '-'*60
    sk_x = sk.sketch(x)
    approx_x = sk.unsketch(sk_x)
    sk.showme(x=x,sk_x=sk_x,approx_x=approx_x)
    sk.compareRows(x,approx_x)

  x = db.onehot(xsym)
  probe(x)
  probe(x.dot(M_edge))

  print '-'*60
  sk_x = sk.sketch(x)
  sk_nx = sk.follow(rel,sk_x)
  approx_nx = sk.unsketch(sk_nx)
  nx = x.dot(M_edge)
  sk.showme(nx=nx, sk_nx=sk_nx, approx_nx=approx_nx)
  sk.compareRows(nx,approx_nx)

