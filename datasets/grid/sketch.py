import sys
import time
import random
import getopt
from tensorlog.helper.countmin_embeddings import Sketcher
from tensorlog import comline,matrixdb

def path_1(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  for pt in zip(range(1,64),range(1,64)):
      start = time.time()
      q = db.onehot("%d,%d"%pt)
      a = q.dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sq)
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
  n = 64*64
  print "native qps",n/native_p
  print "sketch qps",n/sketch_p
    
def path_2(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  for pt in zip(range(1,64),range(1,64)):
      start = time.time()
      q = db.onehot("%d,%d"%pt)
      a = q.dot(M_edge).dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sk.follow(rel,sq))
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
  n = 64*64
  print "native qps",n/native_p
  print "sketch qps",n/sketch_p
    

if __name__ == '__main__':
  if len(sys.argv)<2:
      print "sample usage:"
      print sys.argv[0],"--db 'g64.db|../../datasets/grid/inputs/g64.cfacts'  --x 25,36 --rel edge --k 10 --seed 314159"
      exit(0)
  print 'loading db...'
  optlist,args = getopt.getopt(sys.argv[1:],'x',["db=","x=", "rel=","k=","delta=","seed="])
  optdict = dict(optlist)
  #db = matrixdb.MatrixDB.loadFile('g10.cfacts')
  print 'optdict',optdict
  db = comline.parseDBSpec(optdict.get('--db','g10.cfacts'))
  xsym = optdict.get('--x','5,5')
  rel = optdict.get('--rel','edge')
  k = int(optdict.get('--k','10'))
  delta = float(optdict.get('--delta','0.01'))
  seed = optdict.get('--seed',-1)
  if seed>0: random.seed(seed)
  
  M_edge = db.matEncoding[(rel,2)]
  sk = Sketcher(db,k,delta/db.dim())
  sk.describe()

  path_1(db,sk,M_edge,rel)
  path_2(db,sk,M_edge,rel)
