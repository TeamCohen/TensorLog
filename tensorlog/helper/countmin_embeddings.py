# first draft at code for doing count-min embeddings

import numpy as np
import scipy
try:
  ss = scipy.sparse
except:
  import scipy.sparse as ss
    
import math
import random
import sys
import getopt
import collections

from tensorlog import matrixdb,comline,mutil

LOGBASE = 2
MAX_NUM_HASHES = 1000

def componentwise_min(X,Y):
  """ componentwise_min for two scipy matrices
  - my old version scipy doesn't have csr.minimum(other) implemented!
  """
  X1 = ss.csr_matrix(X)
  X1.data = np.ones_like(X.data)
  Y1 = ss.csr_matrix(Y)
  Y1.data = np.ones_like(Y.data)
  commonIndicesXVals = X.multiply(Y1)
  commonIndicesYVals = Y.multiply(X1)
  data = np.minimum(commonIndicesXVals.data,commonIndicesYVals.data)
  result = ss.csr_matrix((data,commonIndicesXVals.indices,commonIndicesXVals.indptr),shape=X1.shape,dtype='float32')
  return result

class Sketcher(object):

  def __init__(self,db,k,delta,verbose=True):
    """ Follows notation in my notes.
    http://www.cs.cmu.edu/~wcohen/10-605/notes/randomized-algs.pdf

    k = max set size
    delta = prob of any error in unsketch
    t,m = hash functions map to one of t subranges each of width [1...m]
    n = original dimension

    I usually index 0 ... t with d for some reason in this code
    """

    self.db = db
    self.n = db.dim()
    self.delta = delta
    # see comments for unsketch routine
    self.m = int(math.ceil(LOGBASE * k))
    self.t = int(math.ceil(-math.log(delta,LOGBASE)))
    self.hash_salt = [ random.getrandbits(32) for _ in range(MAX_NUM_HASHES) ]
    self.offset = hash("389437892342389")
    self.init_hashmats()

  def init_hashmats(self):
    """ self.hashmat caches out sketch for each object 0...n.  So if x is
    a k-hot vector encoding {i1:w1, ..., ik:wk}, x.dot(hashmat) is the
    countmin sketch obtained by doing add(i1,w1), .... etc

    The countmin sketch is in t parts, each of the parts is a subrange
    of width m, and hashmats[d] is the matrix for the d-th hash
    function, ie it always maps something to the d'th subrange.

    And self.hashmat = sum_d hashmats[d] 
    """
    # self.hashmats[d] is a matrix version of hash_function(d,x)
    self.hashmats = []
    # self.hashmat is sum of self.hashmats - init with an empty csr matrix
    coo_mat = ss.coo_matrix(([],([],[])), shape=(self.n,self.m*self.t))
    self.hashmat = ss.csr_matrix(coo_mat,dtype='float32')
      
    # cache out each hash function
    databuf = np.ones(self.n)
    for d in range(self.t):
      rowbuf = []
      colbuf = []
      for i in range(self.n):
        j = self.hash_function(d,i)
        rowbuf.append(i)
        colbuf.append(j)
      coo_mat = ss.coo_matrix((databuf,(rowbuf,colbuf)), shape=(self.n,self.m*self.t))
      hd = ss.csr_matrix(coo_mat,dtype='float32')
      self.hashmats.append(hd)
      self.hashmat = self.hashmat + hd

  def hash_function(self,d,x): 
    """ The d-th hash function, which maps x into the range [m*d, ..., m*(d+1)]
    """
    return ((hash(x + self.offset)^self.hash_salt[d]) % self.m) + self.m*d

  def sketch(self,X):
    """ Produce a sketch for X
    """
    return X.dot(self.hashmat)

  def unsketch(self,S):
    """ Approximate the matrix that would be sketched as S, i.e., an M so
    that M*hashmat = S.

    The idea is as follows.  Given a sketch S, we can find all the i's
    in the original space that the d-th hash function would map
    to a non-zero index in S using S.hashmats[d].transpose()
    Some of these are noisy and some are from the actual X that was
    hashed into S.

    How much noise is there? The d-th hash function has k bits set,
    call then j1,...,jk. (There could be less, if there were
    collisions in the k-nonzeros of X).  For any particular set bit
    position j, Pr[hash(d,i)=j] = 1/m, where the probability is taken
    over positions i.  So Pr[hash(d,i)=j1 v ... hash(d,i)=jk] <= k/m.

    Let m=2k.  Then Pr[hash(d,i)=j1 v ... hash(d,i)=jk] <= 1/2.  An
    incorrect index i survives in min_d [S.hashmats[d].transpose()]
    must have been noisy in all t of the individual hashes, so
    
    Pr[noisy i in min] <= 2^{-t}
    
    So set t so that 2^{-t} < delta  ==> t > log_2(1/delta)
    """
    result = ss.csr_matrix(S.dot(self.hashmats[0].transpose()))
    for d in range(1,self.t):
      xd = ss.csr_matrix(S.dot(self.hashmats[d].transpose()))
      result = componentwise_min(result,xd)
    return result

  def follow(self,rel,S):
    """ The analog of an operator X.dot(M) in sketch space.  Given S where
    X.dot(hashmat)=S, return the sketch for X.dot(M).
    """
    M = self.db.matEncoding[(rel,2)]
    return self.sketch(self.unsketch(S).dot(M))

  def describe(self):
    print self.__class__.__name__,'n',self.n,'t',self.t,'delta',self.delta,
    print 'm',self.m,'sketch size',self.t*self.m,'compression',float(self.n)/(self.t*self.m)

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
      except ValueError:
        print tag,'=',val
      print

  def compareRows(self,gold,approx,interactive=True):
    """ For debugging, compare gold and approximate vectors to show how
    different they are
    """
    dGold = self.db.rowAsSymbolDict(gold)
    dApprox = self.db.rowAsSymbolDict(approx)
    fp = set(dApprox.keys()) - set(dGold.keys())
    fn = set(dGold.keys()) - set(dApprox.keys())
    if interactive:
      print 'errors:',len(fp),list(fp)[0:10]
      if len(fn)>0:
        print '  also there are false negatives - really?',fn
    else:
      return fp,fn

class Sketcher2(Sketcher):
  """
  Slightly different plan - to follow a sketch S thru a relation
  matrix M, look at the coo_matrix version of M, which is basically
  k_M triples of the form (i,j,w) where i is a row index, j is a
  column index, and w is a weight.  For each M, we construct a
  something like hashmats above, but instead of Hd mapping an entity
  index to [1...m], so Hd[e]=hash(d,e), we let Hd maps an index k from
  [1...k_m], via the row entity i, into [1...m].  Then Hd^T maps S to
  the triples from the coo_matrix for M.  We can 'follow' M by using
  the componentwise_min of the S.Hd^T's to find the non-zero triple
  indices, multiplying by the w's, and then finally using another hash
  to map the j's into sketch space.
  
  This Hd is stored in sketchmatsArg1[rel][d], where rel is the
  functor associated with M.  We also have a second analogous hash
  that maps the j's to sketch space, called sketchmatsArg2[rel].  We
  don't both computing and storing sketchmatArg1[rel] or
  sketchmatsArg2[rel][d], since they are never used.
  """

  def __init__(self,db,k,delta,verbose=True):
    # want to make t 2b * as large
    super(Sketcher2,self).__init__(db,k,delta,verbose)
    # these are like hashmat, and hashmats, but indexed by the functor
    # for a relation
    self.sketchmatsArg1 = collections.defaultdict(list)
    self.sketchmatArg2 = {}
    self.sketchmatWeights = {}
    print 'sketching the db rels....'
    for (functor,arity),M in db.matEncoding.items():
      if arity==2:
        m = scipy.sparse.coo_matrix(M)
        self.sketchmatWeights[functor] = scipy.sparse.csr_matrix(m.data.reshape((1,m.nnz)))
        skShape = (m.nnz,self.m*self.t)  #shape of the sketches we build below
        skRows = np.arange(m.nnz)        #coo.row for sketches we build
        skArg2 = scipy.sparse.coo_matrix(skShape)
        ones = np.ones_like(skRows)
        for d in range(self.t):
          if verbose: print 'hash',d+1,'of',self.t,'for',functor
          else: print ".",
          arrayHasher = np.vectorize(lambda k: self.hash_function(d,k))
          h1d = scipy.sparse.coo_matrix((ones,(skRows,arrayHasher(m.row))),shape=skShape)
          self.sketchmatsArg1[functor].append(scipy.sparse.csr_matrix(h1d))
          h2d = scipy.sparse.coo_matrix((ones,(skRows,arrayHasher(m.col))),shape=skShape)
          skArg2 = skArg2 + h2d
        self.sketchmatArg2[functor] = scipy.sparse.csr_matrix(skArg2)
    print 'done'

  def follow(self,rel,S):
    """ The analog of an operator X.dot(M) in sketch space.  Given S where
    X.dot(hashmat)=S, return the sketch for X.dot(M).
    """
    # nz_indices will be the indices of the coo_matrix for rel where
    # the row# hashes to something in S - unsketch these indices the
    # way we did before
    nz_indices = scipy.sparse.csr_matrix(S.dot(self.sketchmatsArg1[rel][0].transpose()))
    for d in range(1,self.t):
      in_d = scipy.sparse.csr_matrix(S.dot(self.sketchmatsArg1[rel][d].transpose()))
      nz_indices = componentwise_min(nz_indices,in_d)
    # multiply these indices by the corresponding weights, and then
    # move back to sketch space
    return self.sketchmatWeights[rel].multiply(nz_indices).dot(self.sketchmatArg2[rel])

# example: python countmin_embeddings.py --db  'fb15k.db|../../datasets/fb15k-speed/inputs/fb15k-valid.cfacts'  --x m_x_02md_2 --rel american_football_x_football_coach_position_x_coaches_holding_this_position_x_american_football_x_football_historical_coach_position_x_team --k 50
# example: python countmin_embeddings.py --db  'g64.db|../../datasets/grid/inputs/g64.cfacts'  --x 25,36 --rel edge --k 10

if __name__ == "__main__":
  print 'loading db...'
  optlist,args = getopt.getopt(sys.argv[1:],'x',["db=","x=", "rel=","k=","delta=","v=","seed="])
  optdict = dict(optlist)
  #db = matrixdb.MatrixDB.loadFile('g10.cfacts')
  print 'optdict',optdict
  db = comline.parseDBSpec(optdict.get('--db','g10.cfacts'))
  xsym = optdict.get('--x','5,5')
  rel = optdict.get('--rel','edge')
  k = int(optdict.get('--k','10'))
  delta = float(optdict.get('--delta','0.01'))
  v = optdict.get('--v','1')
  seed = optdict.get('--seed',-1)
  if seed>0: random.seed(seed)

  M_edge = db.matEncoding[(rel,2)]
  if v=='1':  
    sk = Sketcher(db,k,delta/db.dim())
  elif v=='2':
    sk = Sketcher2(db,k,delta/db.dim())
  else:
    assert False,'--v should not be %r' % v
  sk.describe()

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
