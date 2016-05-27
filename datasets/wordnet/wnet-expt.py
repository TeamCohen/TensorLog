import sys

import expt
import os.path
import scipy.sparse as SS
import scipy.io
import matrixdb

def uncacheDB(dbFile):
    if not os.path.exists(dbFile):
        print 'creating',dbFile,'...'
        db = matrixdb.MatrixDB.loadFile('wnet.cfacts')
        db.serialize(dbFile)
        print 'created',dbFile
        return db
    else:
        return matrixdb.MatrixDB.deserialize(dbFile)

def uncacheMatPairs(cacheFile,dbFile,trainPred,testPred):
    if True: # not os.path.exists(cacheFile):
        db = uncacheDB(dbFile)
        print 'creating matrix pairs from %s and %s...' % (trainPred,testPred)
        TX,TY = db.matrixAsTrainingData(trainPred,2)
        UX,UY = db.matrixAsTrainingData(testPred,2)
        print 'type(TX)',type(TX)
        print 'created and saving in',cacheFile,'...'
        d = {'tx':TX,'ty':TY,'ux':UX,'uy':UY}
        scipy.io.savemat(cacheFile,d, do_compression=True)
        print 'saved in',cacheFile
        return d
    else: #sort of broken
        d = {}
        scipy.io.loadmat("wnet-%s-XY.mat" % trainPred,d)
        # bleeping scipy.io utils convert from csr_matrix to csc_matrix
        # when you serialize/deserialize, like they think it doesn't
        # make a difference or somethin'
        for k in d.keys():
            if not k.startswith("__"):
                d[k] = d[k].tocsr()
        return d
        
if __name__=="__main__":
    if len(sys.argv)<=1:
        pred = 'hypernym'
    else:
        pred = sys.argv[1]
    print '== pred',pred
    d = uncacheMatPairs('wnet-%s-XY.mat' % pred,'wnet.db','train_i_%s' % pred,'valid_i_%s' % pred)
    params = {'initFiles':["wnet.db","wnet-learned.ppr"],
              'theoryPred':'i_%s' % pred,
              'trainMatPair':(d['tx'],d['ty']),
              'testMatPair':(d['ux'],d['uy']),
              'savedModel':'%s-trained.db' % pred,
              'savedTestPreds':'%s-test.solutions.txt' % pred,
              'savedTrainExamples':'%s-train.examples' % pred,
              'savedTestExamples':'%s-test.examples' % pred,
              'epochs':30,
    }
    expt.Expt(params).run()
