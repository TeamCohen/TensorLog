# not working yet!

import os.path
import scipy.sparse as SS
import scipy.io

import expt
import tensorlog
import matrixdb
import mutil
import ops

def uncacheDB(dbFile):
    if not os.path.exists(dbFile):
        print 'creating',dbFile,'...'
        db = matrixdb.MatrixDB.loadFile('cora.cfacts')
        db.serialize(dbFile)
        print 'created',dbFile
        return db
    else:
        return matrixdb.MatrixDB.deserialize(dbFile)

def uncacheMatPairs(cacheFile,dbFile,exampleFile):
        db = uncacheDB(dbFile)
        print 'preparing examples...'
        d = expt.Expt.propprExamplesAsData(db,exampleFile)
        print 'prepared',d.keys()
        return d
        
if __name__=="__main__":
    ops.OPTIMIZE_COMPONENT_MULTIPLY=False
    dTrain = uncacheMatPairs('cora-XY.mat','cora.db','inputs/train.examples')
    dTest = uncacheMatPairs('cora-XY.mat','cora.db','inputs/test.examples')
    for d in [0,2,3,4,5]:
        prog = tensorlog.ProPPRProgram.load(["cora.db","cora.ppr"])
        prog.db.markAsParam('kaw',1)
        prog.db.markAsParam('ktw',1)
        prog.db.markAsParam('kvw',1)
        prog.maxDepth = d
        prog.setWeights(prog.db.ones())
        params = {'initProgram':prog,
                  'theoryPred':'samebib',
                  'trainMatPair':dTrain['samebib'],
                  'testMatPair':dTest['samebib'],
                  'savedModel':'cora-trained-%d.db' %d,
                  'savedTestPreds':'cora-test-%d.solutions.txt' % d,
                  'savedTrainExamples':'cora-train.examples',
                  'savedTestExamples':'cora-test.examples',
              }
        print 'maxdepth',prog.maxDepth
        expt.Expt(params).run()

