import sys
import time

from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import program
from tensorlog import opfunutil
from tensorlog import expt

def setExptParams():
    print('loading db....')
    db = comline.parseDBSpec("tmp-cache/fb15k.db|inputs/fb15k-valid.cfacts")
    print('loading program....')
    prog = comline.parseProgSpec("inputs/fb15k.ppr",db)
    print('loading queries....')
    queries = fbQueries(prog,db)
    modeSet = set(mode for (mode,_) in queries)
    return (db,prog,modeSet,queries)

def compileAll(db,prog,modeSet,queries):
    start = time.time()
    k = 0
    for mode in modeSet:
        if prog.findPredDef(mode):
            k += 1
            fun = prog.compile(mode)
    fps = k/(time.time() - start)
    print("compiled",k,"of",len(modeSet),"functions at",fps,"fps")
    return fps

def runNative(db,prog,modeSet,queries):
    dset = comline.parseDatasetSpec('tmp-cache/fb15k-valid.dset|inputs/fb15k-valid.examples',db)
    #dataset.Dataset.loadProPPRExamples(db,'inputs/fb15k-valid.examples')
    start = time.time()
    for mode in dset.modesToLearn():
        if prog.findPredDef(mode):
            X = dset.getX(mode)
            fun = prog.function[(mode, 0)]
            fun.eval(db, [X], opfunutil.Scratchpad())
    qps = len(queries)/(time.time() - start)
    print("answered",len(queries),"queries at",qps,"qps")
    return qps

def runSequential(db,prog,modeSet,queries):
    start = time.time()
    k = 0
    for (mode,vx) in queries:
        fun = prog.function[(mode,0)]
        fun.eval(db, [vx], opfunutil.Scratchpad())
        k += 1
        if not k%100: print("answered",k,"queries")
    qps = len(queries)/(time.time() - start)
    print("answered",len(queries),"queries at",qps,"qps")
    return qps

def fbQueries(prog,db):
  queries = []
  ignored = 0
  for line in open("inputs/fb15k-valid.examples"):
      k1 = line.find("(")
      k2 = line.find(",")
      pred = line[:k1]
      x = line[k1+1:k2]
      mode = declare.asMode("%s/io" % pred)
      if prog.findPredDef(mode):
          vx = db.onehot(x)
          queries.append((mode, vx))
      else:
          ignored += 1
  print(len(queries), "queries loaded", "ignored", ignored)
  return queries


def runMain():
    (db,prog,modeSet,queries) = setExptParams()
    fps = compileAll(db,prog,modeSet,queries)
    qps1 = runSequential(db,prog,modeSet,queries)
    qps2 = runNative(db,prog,modeSet,queries)
    return (fps,qps1,qps2)

def runCross():
    (db,prog,modeSet,queries) = setExptParams()
    from tensorlog import xctargets
    CROSSCOMPILERS = []
    CROSSLEARNERS = {}
    if xctargets.theano:
      from tensorlog import theanoxcomp
      for c in [
        #theanoxcomp.DenseMatDenseMsgCrossCompiler,
        theanoxcomp.SparseMatDenseMsgCrossCompiler
        ]:
        CROSSCOMPILERS.append(c)
        CROSSLEARNERS[c]=theanoxcomp.FixedRateGDLearner
    if xctargets.tf:
      from tensorlog import tensorflowxcomp
      for c in [
        #tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
        tensorflowxcomp.SparseMatDenseMsgCrossCompiler,
        ]:
        CROSSCOMPILERS.append(c)
        CROSSLEARNERS[c]=tensorflowxcomp.FixedRateGDLearner
    results = {}
    for compilerClass in CROSSCOMPILERS:
        xc = compilerClass(prog)
        print(expt.fulltype(xc))
        
        # compileAll
        start = time.time()
        k = 0
        # compile
        for mode in modeSet:
            if not prog.findPredDef(mode):continue
            k += 1
            xc.ensureCompiled(mode)
        fps = k / (time.time() - start)
        print("compiled",k,"of",len(modeSet),"functions at",fps,"fps")
        
        # runSequential
        start = time.time()
        k = 0
        for (mode,vx) in queries:
            xc.inferenceFunction(mode)(vx)
            k += 1
            if not k%100: print("answered",k,"queries")
        qps1 = len(queries) / (time.time() - start)
        print("answered",len(queries),"queries at",qps1,"qps")
        
        # runNative
        dset = comline.parseDatasetSpec('tmp-cache/fb15k-valid.dset|inputs/fb15k-valid.examples',db)
        start = time.time()
        for mode in dset.modesToLearn():
            if not prog.findPredDef(mode):continue
            X = dset.getX(mode)
            xc.inferenceFunction(mode)(X)
        qps2 = len(queries) / (time.time() - start)
        print("answered",len(queries),"queries at",qps2,"qps")
        results[expt.fulltype(xc)] = (fps,qps1,qps2)
    return results

if __name__ == "__main__":
    fps,qps1,qps2 = runMain()
    if "cross" in sys.argv[1:]: runCross()
