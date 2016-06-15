# status: 
# underflow on mode i_base_x_aareas_x_schema_x_administrative_area_x_administrative_children(i,o)


import sys

import exptv2
import dataset
import tensorlog
import os.path
import scipy.sparse as SS
import scipy.io
import matrixdb

def uncacheDB(dbFile):
    if not os.path.exists(dbFile):
        print 'creating',dbFile,'...'
        db = matrixdb.MatrixDB.loadFile('fb.cfacts')
        db.serialize(dbFile)
        print 'created',dbFile
        return db
    else:
        return matrixdb.MatrixDB.deserialize(dbFile)

def uncacheMatPairs(cacheFile,dbFile,trainPred,testPred):
    if not os.path.exists(cacheFile):
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
    else:
        d = {}
        scipy.io.loadmat("fb-XY.mat",d)
        # bleeping scipy.io utils convert from csr_matrix to csc_matrix
        # when you serialize/deserialize, like they think it doesn't
        # make a difference or somethin'
        for k in d.keys():
            if not k.startswith("__"):
                d[k] = d[k].tocsr()
        return d
        
if __name__=="__main__":
    params = {}
    pred = 'all'
    if len(sys.argv)>1:
        pred = sys.argv[1]
        params['targetPred'] = 'train_i_%s/io' % pred
        print '== pred',pred
    db = matrixdb.MatrixDB.uncache('tmp-cache/fb15k.db','fb.cfacts')
    trainData = dataset.Dataset.uncacheExamples('tmp-cache/fb15k-train.dset',db,'inputs/train.examples')
    testData = dataset.Dataset.uncacheExamples('tmp-cache/fb15k-test.dst',db,'inputs/valid.examples')
    print 'train:','\n  '.join(trainData.pprint()[:10]),'\n...'
    print 'test: ','\n  '.join(testData.pprint()[:10]),'\n...'
    prog = tensorlog.ProPPRProgram.load(['fb-learned.ppr'],db=db)
    prog.setWeights(db.ones())
    
    #d = uncacheMatPairs('fb-%s-XY.mat' % pred,'fb.db','train_i_%s' % pred,'valid_i_%s' % pred)
    params.update({'initProgram':prog,
                   'trainData':trainData, 'testData':testData,
                   'savedModel':'%s-trained.db' % pred,
                   'savedTestPreds':'%s-test.solutions.txt' % pred,
                   'savedTrainExamples':'%s-train.examples' % pred,
                   'savedTestExamples':'%s-test.examples' % pred,
                   'epochs':30,
    })
    exptv2.Expt(params).run()
