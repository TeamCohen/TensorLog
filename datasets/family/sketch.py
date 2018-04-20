import sys
import time
import random
import getopt
from tensorlog.helper.countmin_embeddings import Sketcher,Sketcher2
from tensorlog.helper.fast_sketch import FastSketcher
from tensorlog import comline,matrixdb

def path_1(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  n = db.dim()
  for x in range(n):
      start = time.time()
      xsymb = db.asSymbol(x)
      if xsymb == None: continue
      q = db.onehot(xsymb)
      a = q.dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sq)
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
  print "native qps",n/native_p
  print "sketch qps",n/sketch_p
    
def path_2(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  n = db.dim()
  for x in range(n):
      start = time.time()
      xsymb = db.asSymbol(x)
      if xsymb == None: continue
      q = db.onehot(xsymb)
      a = q.dot(M_edge).dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sk.follow(rel,sq))
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
  print "native qps",n/native_p
  print "sketch qps",n/sketch_p
  
if __name__ == '__main__':
  if len(sys.argv)<2:
      print "sample usage:"
      print sys.argv[0],"--db 'kin.db|inputs/kinship.cfacts' --rel father --k 10 --seed 314159"
      exit(0)
  print 'loading db...'
  optlist,args = getopt.getopt(sys.argv[1:],'x',["db=","x=", "rel=","k=","delta=","seed="])
  optdict = dict(optlist)
  #db = matrixdb.MatrixDB.loadFile('g10.cfacts')
  print 'optdict',optdict
  db = comline.parseDBSpec(optdict.get('--db','g10.cfacts'))
  print "nnz",db.size()
  rel = optdict.get('--rel','father')
  k = int(optdict.get('--k','10'))
  delta = float(optdict.get('--delta','0.01'))
  seed = optdict.get('--seed',-1)
  if seed>0: random.seed(seed)
  
  M_edge = db.matEncoding[(rel,2)]

  for sclass in [Sketcher2,Sketcher,FastSketcher]:
      start = time.time()
      sk = sclass(db,k,delta/db.dim())
      sk.describe()
      print "load",time.time()-start,"sec"
      path_1(db,sk,M_edge,rel)
      path_2(db,sk,M_edge,rel)
