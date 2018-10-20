import sys
import time

from tensorlog import comline
from tensorlog import declare
from tensorlog import interp
from tensorlog import mutil

BATCHSIZE=250

def setExptParams(n):
    factFile = 'smoker-%d.cfacts' % n
    queryFile = 'query-entities-%d.txt' % n
    optdict,args = comline.parseCommandLine(['--prog','smokers.ppr','--proppr','--db',factFile])
    ti = interp.Interp(optdict['prog'])
    ti.prog.setRuleWeights()
    ti.prog.maxDepth = 99
    rows = []
    for line in open(queryFile):
        sym = line.strip()
        rows.append(ti.db.onehot(sym))
        if len(rows)==BATCHSIZE:
            break
    X = mutil.stack(rows)
    print(len(rows),'queries')
    return ti,X,rows

def runMain(n,minibatch):

    (ti,X,rows) = setExptParams(n)
    nQueries = X.shape[0]
    start0 = time.time()
    for modeString in ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]:
        print('eval',modeString, end=' ')
        start = time.time()
        if minibatch:
            ti.prog.eval(declare.asMode(modeString), [X])
        else:
            for Xi in rows:
                ti.prog.eval(declare.asMode(modeString), [Xi])
        print('time',time.time() - start,'sec')
    tot = time.time() - start0
    print('batch size',len(rows))
    print('minibatch',minibatch)
    print('total query time',tot,'sec')
    print('queries/sec',nQueries/tot)
    print('%.2f\t%.2f' % (tot,nQueries/tot))
    return tot

# usage n [no-minibatch]
if __name__=="__main__":
    n = 100
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    minibatch = True
    if len(sys.argv) > 2:
        minibatch = False

    t = runMain(n,minibatch)
    print('time',t)
