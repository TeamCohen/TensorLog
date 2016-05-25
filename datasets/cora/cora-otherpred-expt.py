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
        
def coraExpt(pred):
    print '== learning predicate',pred
    ops.OPTIMIZE_COMPONENT_MULTIPLY=False
    dTrain = uncacheMatPairs('cora-XY.mat','cora.db','inputs/train.examples')
    dTest = uncacheMatPairs('cora-XY.mat','cora.db','inputs/test.examples')
    prog = tensorlog.ProPPRProgram.load(["cora.db","cora.ppr"])
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    prog.maxDepth = 1
    prog.setWeights(prog.db.ones())
    params = {'initProgram':prog,
              'theoryPred':pred,
              'trainMatPair':dTrain[pred],
              'testMatPair':dTest[pred],
              'savedModel':'cora-%s-trained.db' % pred,
              'savedTestPreds':'cora-%s-test.solutions.txt' % pred,
              'savedTrainExamples':'cora-%s-train.examples' % pred,
              'savedTestExamples':'cora-%s-test.examples' %pred,
    }
    print 'maxdepth',prog.maxDepth
    expt.Expt(params).run()

if __name__=="__main__":
    coraExpt('sametitle')
    coraExpt('samevenue')
    coraExpt('sameauthor')


