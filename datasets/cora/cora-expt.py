import os.path
import scipy.sparse as SS
import scipy.io

import exptv1
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
        d = exptv1.Expt.propprExamplesAsData(db,exampleFile)
        print 'prepared',d.keys()
        return d
        
if __name__=="__main__":
    ops.conf.optimize_component_multiply=False
    dTrain = uncacheMatPairs('cora-XY.mat','cora.db','inputs/train.examples')
    dTest = uncacheMatPairs('cora-XY.mat','cora.db','inputs/test.examples')
    prog = tensorlog.ProPPRProgram.load(["cora.db","cora.ppr"])
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 1
    prog.setWeights(prog.db.ones())
    params = {'initProgram':prog,
              'theoryPred':'samebib',
              'trainMatPair':dTrain['samebib'],
              'testMatPair':dTest['samebib'],
              'savedModel':'cora-trained.db',
              'savedTestPreds':'cora-test.solutions.txt',
              'savedTrainExamples':'cora-train.examples',
              'savedTestExamples':'cora-test.examples',
    }
    print 'maxdepth',prog.maxDepth
    exptv1.Expt(params).run()
