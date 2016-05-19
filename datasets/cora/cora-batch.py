# current failure mode: inconsistent shapes computing Y-P

import os.path
import scipy.sparse as SS
import scipy.io

import expt
import tensorlog
import matrixdb
import mutil
import ops 
import funs
ops.TRACE=False
funs.TRACE=False

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
    dTrain = uncacheMatPairs('cora-XY.mat','cora.db','raw/train.examples')
    dTest = uncacheMatPairs('cora-XY.mat','cora.db','raw/test.examples')
    prog = tensorlog.ProPPRProgram.load(["cora.db","cora.ppr"])
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.setWeights(prog.db.ones())
    params = {'initProgram':prog,
              #'theoryPred':'samebib',
              'trainData':dTrain,
              'testData':dTest,
              'savedModel':'cora-batch-trained.db',
              'savedTestPreds':'cora-batch-test.solutions.txt',
              'savedTrainExamples':'cora-batch-train.examples',
              'savedTestExamples':'cora-batch-test.examples',
    }
    expt.BatchExpt(params,{'epochs':5}).run()
