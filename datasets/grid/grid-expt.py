import sys
import time

import tensorlog
import exptv1
import declare
import matrixdb

if __name__=="__main__":
    
    gridFile = sys.argv[1]
    startNode = sys.argv[2]
    matrixdb.conf.allow_weighted_tuples = True

    for d in [4,8,16,32,64,99]:
        print 'depth',d,
        ti = tensorlog.Interp(initFiles=[gridFile,'grid.ppr'],proppr=False)
        ti.prog.maxDepth = d
        start = time.time()
        ti.prog.evalSymbols(ti._asMode("path/io"), [startNode])
        print 'time',time.time() - start,'sec'
