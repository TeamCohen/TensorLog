import os.path
import scipy.sparse as SS
import scipy.io

import expt
import tensorlog
import matrixdb
import mutil
import logging
import funs

#logging.basicConfig(level=logging.DEBUG)

def uncacheDB(dbFile):
    if not os.path.exists(dbFile):
        print 'creating',dbFile,'...'
        db = matrixdb.MatrixDB.loadFile('cora.cfacts')
        db.serialize(dbFile)
        print 'created',dbFile
        return db
    else:
        return matrixdb.MatrixDB.deserialize(dbFile)

def uncacheMatPairs(dbFile,exampleFile):
        db = uncacheDB(dbFile)
        print 'preparing examples...'
        d = expt.Expt.propprExamplesAsData(db,exampleFile)
        print 'prepared',d.keys()
        return d
        
if __name__=="__main__":
    stem = "top-1000-near-google"
    dTrain = uncacheMatPairs('%s.db' % stem,'raw/%s.train.examples' % stem)
    dTest = uncacheMatPairs('%s.db' % stem,'raw/%s.test.examples' % stem)
    prog = tensorlog.ProPPRProgram.load(["%s.db" % stem,"%s-recursive.ppr" % stem])
    prog.setWeights(prog.db.ones())
    prog.maxDepth=4
    params = {'initProgram':prog,
              #'theoryPred':'concept_atdate',
              'trainData':dTrain,
              'testData':dTest,
              'savedModel':'%s-trained.db' % stem,
              'savedTestPreds':'%s-test.solutions.txt' % stem,
              'savedTrainExamples':'%s-train.examples' % stem,
              'savedTestExamples':'%s-test.examples' % stem,
    }
    expt.BatchExpt(params,{'epochs':5}).run()
