import tensorlog
import expt
import learn

if __name__=="__main__":

    db = tensorlog.parseDBSpec('tmp-cache/cora-linear.db|inputs/cora.cfacts')
    trainData = tensorlog.parseDatasetSpec('tmp-cache/cora-linear-train.dset|inputs/train.examples', db)
    testData = tensorlog.parseDatasetSpec('tmp-cache/cora-linear-test.dset|inputs/test.examples', db)
    prog = tensorlog.parseProgSpec("cora-linear.ppr",db,proppr=True)
    prog.setWeights(db.ones())
    prog.db.markAsParam('kaw',1)
    prog.db.markAsParam('ktw',1)
    prog.db.markAsParam('kvw',1)
    learner = learn.FixedRateGDLearner(prog,regularizer=learn.L2Regularizer(),traceFun=learn.Learner.cheapTraceFun,epochs=5)
    prog.maxDepth = 3
    params = {'prog':prog,
              'trainData':trainData, 'testData':testData,
              'targetMode':'samebib/io',
              'savedModel':'tmp-cache/cora-trained.db',
              'savedTestPredictions':'tmp-cache/cora-test.solutions.txt',
              'savedTrainExamples':'tmp-cache/cora-train.examples',
              'savedTestExamples':'tmp-cache/cora-test.examples',
              'learner':learner
    }
    print 'maxdepth',prog.maxDepth
    expt.Expt(params).run()
