# status: 
# underflow on training epoch 2 mode i_base_x_aareas_x_schema_x_administrative_area_x_administrative_children(i,o)


import sys

import exptv2
import dataset
import tensorlog
import os.path
import scipy.sparse as SS
import scipy.io
import matrixdb
        
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
    
    params.update({'initProgram':prog,
                   'trainData':trainData, 'testData':testData,
                   'savedModel':'%s-trained.db' % pred,
                   'savedTestPreds':'%s-test.solutions.txt' % pred,
                   'savedTrainExamples':'%s-train.examples' % pred,
                   'savedTestExamples':'%s-test.examples' % pred,
                   'epochs':10,
    })
    exptv2.Expt(params).run()
