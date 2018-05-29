# first draft at code for doing count-min embeddings

import collections
import numpy as np

def embedder_matrix(original_dim,embedded_dim,hash_salt):
  num_hashes = len(hash_salt)
  def hash_function(d,x): offset = hash("pittsburgh"); return (hash(x+offset)^hash_salt[d]) % embedded_dim
  h_rows = []
  for i in range(original_dim):
    row = [0.0] * embedded_dim
    for d in range(num_hashes):
      j = hash_function(d,i)
      row[j] = 1.0
    h_rows.append(row)
  return np.array(h_rows)

def sample_matrix(original_dim):
  m = np.zeros(shape=(original_dim,original_dim))
  m[0,1] = m[0,3] = 1.0
  m[1,0] = m[1,2] = 1.0
  m[2,1] = m[2,5] = 1.0
  m[3,0] = m[3,4] = 1.0
  m[4,3] = m[4,7] = 1.0
  m[5,2] = m[5,8] = 1.0
  m[6,7] = 1.0
  m[7,6] = m[7,4] = m[7,8] = 1.0
  m[8,7] = m[8,5] = 1.0
  return m

def show(label,mat,code=None,h=None):
  print '=' * 10, label, 'shape', mat.shape, '=' * 10
  print mat
  if code=='onehot':
    return pp_decode_onehot(mat)
  elif code=='embedded':
    return pp_decode_embedded(mat,h)

def pp_decode_embedded(mat,h):
  n_rows_m,n_cols_m = mat.shape
  n_rows_h,n_cols_h = h.shape
  assert n_cols_m==n_cols_h
  result = collections.defaultdict(set)
  for r1 in range(n_rows_m):
    print 'row',r1,'contains embedding:',
    for r2 in range(n_rows_h):
      if np.all(mat[r1,:]>=h[r2,:]):
        print r2,
        result[r1].add(r2)
    print
  return result

def pp_decode_onehot(mat):
  n_rows,n_cols = mat.shape
  result = collections.defaultdict(set)
  for r in range(n_rows):
    print 'row',r,'contains:',
    for c in range(n_cols):
      if mat[r,c]!=0:
        print c,
        result[r].add(c)
    print
  return result

def onehot(i,original_dim):
  v = np.zeros(shape=(1,original_dim))
  v[0,i] = 1.0
  return v

# summary:
#
# let N be original space, M be embedding space
# H (h in code) maps k-hot vectors to CM embeddings:
#   for all i, H[i,j_k]=1 for D distinct hashes of i, { j_1, ..., j_K }
#   i.e., the J-th column of H indicates which indices [in the original N space] get hashed to index J in the c-m space
#
# m is a N-by-N matrix, intended to encode a relation p(X,Y)
#
# 1) to embed a one-hot vector v, compute ev = vH
# 2) to embed a matrix M mapping i to i' in the original space,
#    let H1 be a row-normalized version of H, then compute eM = H1^T M H
#    Then, absent collisions, ev eM ~= (vM) H
# 3) to see if a row v of an embedded matrix contains i,
#    test if np.all( v>= u_iH ), where u_i is one-hot for i
# 4) to estimate (vM)[i,i1] from w = (ev eM), look at
#    min{ w[ w >= (u_i1)H ] }  ---I think, not tested

def run_main1():
  original_dim = 10
  embedded_dim = 5

  #hash_salt = [hash("william"),hash("cohen"),hash("rubber duckie")]
  hash_salt = [hash("william"),hash("cohen")]
  h = embedder_matrix(original_dim,embedded_dim,hash_salt)
  show('h',h)
  m = sample_matrix(original_dim)
  show('m',m,code='onehot')
  mh = np.dot(m,h)
  show('mh',mh,code='embedded',h=h)
  #this isn't quite right since you need to allow for possibility
  #of hash collisions in h
  oneByD = np.reciprocal(h.sum(1))
  hTbyD =  h.transpose()*oneByD
  show('h^T/D',hTbyD)
  E_m = np.dot(hTbyD,mh)
  show('E_m',E_m)

  def check_results(i):
    ui = onehot(i,original_dim)
    ui_m = np.dot(ui,m)
    baseline = pp_decode_onehot(ui_m)
    E_ui = np.dot(ui,h)
    E_ui_dot_E_m = np.dot(E_ui,E_m)
    proposed = pp_decode_embedded(E_ui_dot_E_m,h)
    n = collisions = 0
    for i in baseline:
      assert i in proposed
      for j in baseline[i]:
        assert j in proposed[i]
        n += 1
      for j in proposed[i]:
        if j not in baseline[i]:
          collisions += 1
    print 'row',i,'collisions',collisions,'baseline',baseline,'proposed',proposed
    return collisions,n

  tot = tot_collisions = 0
  for i in range(original_dim):
    c,n = check_results(i)
    tot_collisions += c
    tot += n
  print 'tot_collisions',tot_collisions,'tot',tot

if __name__ == "__main__":
  original_dim = 10
  embedded_dim = 5
  hash_salt = [hash("william"),hash("cohen")]
  H = embedder_matrix(original_dim,embedded_dim,hash_salt)
  x = onehot(7,original_dim)
  ex = np.dot(x,H)
  print 'x',x
  print 'ex',ex


  
