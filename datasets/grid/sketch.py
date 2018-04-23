import sys
import time
import random
import getopt
from tensorlog.helper.countmin_embeddings import Sketcher,Sketcher2
from tensorlog.helper.fast_sketch import FastSketcher,FastSketcher2
from tensorlog import comline,matrixdb

def path_1(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  k=0
  errors=[]
  for pt in zip(range(1,64),range(1,64)):
      k+=1
      start = time.time()
      q = db.onehot("%d,%d"%pt)
      a = q.dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sq)
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
      fp,fn = sk.compareRows(a,sk.unsketch(sa),interactive=False)
      errors.append( (len(fp),len(fn)) )
  print "native qps",k/native_p
  print "sketch qps",k/sketch_p
  fp = [p for p,n in errors]
  fn = [n for p,n in errors]
  print "min / avg / max fp:",min(fp),sum(fp)/k,max(fp)
  print "#fn > 0:",sum(1 if n>0 else 0 for n in fn )
    
def path_2(db,sk,M_edge,rel):
  native_p = 0
  sketch_p = 0
  k=0
  errors=[]
  for pt in zip(range(1,64),range(1,64)):
      k+=1
      start = time.time()
      q = db.onehot("%d,%d"%pt)
      a = q.dot(M_edge).dot(M_edge)
      native_t = time.time()
      sq = sk.sketch(q)
      sa = sk.follow(rel,sk.follow(rel,sq))
      sketch_t = time.time()
      native_p += native_t-start
      sketch_p += sketch_t - native_t
      fp,fn = sk.compareRows(a,sk.unsketch(sa),interactive=False)
      errors.append( (len(fp),len(fn)) )
  print "native qps",k/native_p
  print "sketch qps",k/sketch_p
  fp = [p for p,n in errors]
  fn = [n for p,n in errors]
  print "min / avg / max fp:",min(fp),sum(fp)/k,max(fp)
  print "#fn > 0:",sum(1 if n>0 else 0 for n in fn )

defaults = 'g64.db|inputs/g64.cfacts 50'.split()
if __name__ == '__main__':
  if len(sys.argv)<2:
      print "sample usage:"
      print sys.argv[0],"--db '%s'  --x 25,36 --rel edge --k %s --seed 314159" % tuple(defaults)
      exit(0)
  print 'loading db...'
  optlist,args = getopt.getopt(sys.argv[1:],'x',["db=","x=", "rel=","k=","delta=","seed="])
  optdict = dict(optlist)
  #db = matrixdb.MatrixDB.loadFile('g10.cfacts')
  print 'optdict',optdict
  db = comline.parseDBSpec(optdict.get('--db',defaults[0]))
  xsym = optdict.get('--x','5,5')
  rel = optdict.get('--rel','edge')
  k = int(optdict.get('--k',defaults[1]))
  delta = float(optdict.get('--delta','0.01'))
  seed = optdict.get('--seed',-1)
  
  M_edge = db.matEncoding[(rel,2)]
  for sclass in [FastSketcher2,
                 FastSketcher,
                 Sketcher2,
                 Sketcher]:

      if seed>0: random.seed(seed)
      sk = sclass(db,k,delta/db.dim(),verbose=False)
      sk.describe()

      path_1(db,sk,M_edge,rel)
      path_2(db,sk,M_edge,rel)
