import sys
import time
import logging
from tensorlog import comline
from tensorlog import declare
from tensorlog import interp
from tensorlog import mutil
from tensorlog import expt
from tensorlog import xctargets

CROSSCOMPILERS = []
if xctargets.theano:
  from tensorlog import theanoxcomp
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    CROSSCOMPILERS.append(c)


modes = ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]
def setExptParams():
    optdict,args = comline.parseCommandLine('--prog smokers.ppr --proppr --db smokers.cfacts'.split())
    ti = interp.Interp(optdict['prog'])
    ti.prog.setRuleWeights()
    ti.prog.maxDepth = 99
    rows = []
    for line in open('query-entities.txt'):
        sym = line.strip()
        rows.append(ti.db.onehot(sym))
    X = mutil.stack(rows)
    return ti,X

def runMain():

    (ti,X) = setExptParams()
    start0 = time.time()
    
    for modeString in modes:
        print('eval',modeString, end=' ')
        start = time.time()
        ti.prog.eval(declare.asMode(modeString), [X])
        print('time',time.time() - start,'sec')
    tot = time.time() - start0
    print('total time',tot,'sec')
    return tot

if __name__=="__main__":
    t = runMain()
    print('time',t)

    (ti,X) = setExptParams()
    for compilerClass in CROSSCOMPILERS:
        start0=time.time()
        xc = compilerClass(ti.prog)
        print(expt.fulltype(xc))
        # compile everything
        for modeString in modes:
            mode = declare.asMode(modeString)
            xc.ensureCompiled(mode)
            print('eval',modeString, end=' ')
            start = time.time()
            xc.inferenceFunction(mode)(X)
            print('time',time.time() - start,'sec')
        print('total time',expt.fulltype(xc),time.time()-start0,'sec')

