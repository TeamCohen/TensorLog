# current failure mode: inconsistent shapes computing Y-P

import os.path
import scipy.sparse as SS
import scipy.io

import expt
import tensorlog
import matrixdb
import mutil

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
#    if not os.path.exists(cacheFile):
        db = uncacheDB(dbFile)
#        print 'creating matrix pairs from %s...' % exampleFile
        print 'preparing examples...'
        d = expt.Expt.propprExamplesAsData(db,exampleFile)
        print 'prepared',d.keys()
#        print 'd.keys()',d.keys()
#        print 'd.values',map(type,d.values())
#        scipy.io.savemat(cacheFile,d, do_compression=True)
#        print 'saved in',cacheFile
        return d
#    else:
#        d = {}
#        scipy.io.loadmat(cacheFile,d)
#        # bleeping scipy.io utils convert from csr_matrix to csc_matrix
#        # when you serialize/deserialize, like they think it doesn't
#        # make a difference or somethin'
#        for k in d.keys():
#            if not k.startswith("__"):
#                print 'k',k,'type',type(d[k])
#                (X,Y) = d[k]
#                d[k] = (X.tocsr(),Y.tocsr())
#                print 'loading matrix',k,mutil.numRows(d[k]),'rows',d[k].nnz,'non-zeros'
#        return d
        
if __name__=="__main__":
#    d = uncacheMatPairs('cora-XY.mat','cora.db','train_samebib','test_samebib')
    dTrain = uncacheMatPairs('cora-XY.mat','cora.db','raw/train.examples')
    dTest = uncacheMatPairs('cora-XY.mat','cora.db','raw/test.examples')
    prog = tensorlog.ProPPRProgram.load(["cora.db","cora.ppr"])
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
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
    expt.Expt(params).run()
