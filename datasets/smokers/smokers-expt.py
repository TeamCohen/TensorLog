import sys
import time

import tensorlog
import expt
import declare
import mutil

if __name__=="__main__":
    
    ti = tensorlog.Interp(initFiles=['smokers.cfacts','smokers.ppr'])
    ti.prog.maxDepth = 99
    rows = []
    for line in open('query-entities.txt'):
        sym = line.strip()
        rows.append(ti.db.onehot(sym))
    X = mutil.stack(rows)

    start0 = time.time()
    for modeString in ["t_stress/io", "t_influences/io","t_cancer_spont/io", "t_cancer_smoke/io"]:
        print 'eval',modeString,
        start = time.time()
        ti.prog.eval(declare.asMode(modeString), [X])
        print 'time',time.time() - start,'sec'
    print 'total time', time.time() - start0,'sec'
#    ti.list("t_influences/io")
#    ti.prog.eval(ti._asMode("t_influences/io"), [X])


