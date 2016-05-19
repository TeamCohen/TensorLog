import sys
import time

import tensorlog
import expt
import declare

if __name__=="__main__":
    
    gridFile = sys.argv[1]
    startNode = sys.argv[2]

    for d in [0,1,2,3]:
        ti = tensorlog.Interp(initFiles=[gridFile,'grid.ppr'],proppr=False)
        ti.prog.maxDepth = d
        P = ti.prog.evalSymbols(ti._asMode("path/io"), [startNode])
        print 'd',d,ti.db.matrixAsSymbolDict(P)

    sys.exit(-1)

    for d in [4,8,16,32,64,99]:
        print 'depth',d,
        ti = tensorlog.Interp(initFiles=[gridFile,'grid.ppr'],proppr=False)
        ti.prog.maxDepth = d
        start = time.time()
        ti.prog.evalSymbols(ti._asMode("path/io"), [startNode])
        print 'time',time.time() - start,'sec'
