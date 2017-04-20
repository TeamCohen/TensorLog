import sys
import time

from tensorlog import comline
from tensorlog import program
from tensorlog import expt
from tensorlog import declare
from tensorlog import mutil
from tensorlog import xctargets

CROSSCOMPILERS = []
if xctargets.theano:
  from tensorlog import theanoxcomp
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    CROSSCOMPILERS.append(c)
if xctargets.tf:
  from tensorlog import tensorflowxcomp
  for c in [
    tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
    tensorflowxcomp.SparseMatDenseMsgCrossCompiler,
    ]:
    CROSSCOMPILERS.append(c)

if __name__=="__main__":

    optdict,args = comline.parseCommandLine('--prog smokers.ppr --proppr --db smokers.cfacts'.split())
    prog = optdict['prog']
    prog.setRuleWeights()
    prog.maxDepth = 99
    rows = []
    for line in open('query-entities.txt'):
        sym = line.strip()
        rows.append(prog.db.onehot(sym))
    X = mutil.stack(rows)

    start0 = time.time()
    modes = ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]
    for modeString in modes:
        print 'eval',modeString,
        start = time.time()
        prog.eval(declare.asMode(modeString), [X])
        print 'time',time.time() - start,'sec'
    print 'total time', time.time() - start0,'sec'

    
    for compilerClass in CROSSCOMPILERS:
        print compilerClass.__name__
        start0=time.time()
        xc = compilerClass(prog)
        # compile everything
        for modeString in modes:
            mode = declare.asMode(modeString)
            xc.ensureCompiled(mode)
            print 'eval',modeString,compilerClass.__name__
            start = time.time()
            xc.inferenceFunction(mode)(X)
            print 'time',time.time() - start,'sec'
        print 'total time',compilerClass.__name__,time.time()-start0,'sec'
