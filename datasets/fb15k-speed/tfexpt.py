import sys
import time
import tensorflow as tf

import expt

from tensorlog import simple
from tensorlog import declare

def tfCompileAll(tlog,modeSet,queries):
  t0 = time.time()
  k = 0
  print 'compiling',len(modeSet),'modes'
  for mode in modeSet:
    if tlog.prog.findPredDef(mode):
      k += 1
      _ = tlog.inference(mode)
      if k%20 == 0:
        sys.stderr.write('compiled %d functions in %f.3 sec\n' % (k,time.time()-t0))
  t1 = time.time()
  fps = k/(t1-t0)
  print 'tlog compiled',k,'functions at',fps,'fps'
  return fps

def runTF(tlog):
  dset = tlog.load_small_dataset('inputs/fb15k-valid.examples')
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  t0 = time.time()
  k = 0
  for mode in dset:
    if tlog.prog.findPredDef(declare.asMode(mode)):
      (X,Y) = dset[mode]
      f = tlog.inference(mode)
      session.run(f, feed_dict={tlog.input_placeholder_name(mode):X})
      k += X.shape[0]
  t1 = time.time()
  qps = k/(t1-t0)
  print 'tlog executes on',k,'inputs at',qps,'qps'
  return qps

def runMain():
  (db,prog,modeSet,queries) = expt.setExptParams()
  tlog = simple.Compiler(db=db, prog=prog, autoset_db_params=False)
  fps1 = expt.compileAll(db,prog,modeSet,queries)
  fps2 = tfCompileAll(tlog,modeSet,queries) # expect <= 2.5 fps
  qps = runTF(tlog) # expect less than 23 qps
  return fps2,qps

if __name__ == "__main__":
  fps,qps = runMain()
