# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for running experiments
#

import time
import collections

import dataset
import matrixdb
import tensorlog
import declare
import learn
import mutil
import config

conf = config.Config()

conf.help.num_train_predictions_shown = 'Number of training-data predictions to display'
conf.num_train_predictions_shown = 0

class Expt(object):

    def __init__(self,configDict):
        self.config = configDict

    def run(self):
        self._run(**self.config)

    def _run(self,
             initProgram=None,trainData=None, testData=None, targetPred=None, epochs=5,
             savedTestPreds=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None):

        """ Run an experiment, given a whole bunch of parameters.
        savedTestPreds, savedTestExamples, savedTrainExamples: if not None, then
        serialize predictions and examples for later eval with ProPPR tools.
        savedModel: save result of training somewhere
        """

        if targetPred: targetPred = declare.asMode(targetPred)

        ti = tensorlog.Interp(initProgram=initProgram)

        tmodes = trainData.modesToLearn()
        if targetPred==None: assert len(tmodes)==1,'multipredicate training not implemented'
        else: assert targetPred in tmodes, 'target predicate %r not in training data' % targetPred
        mode = targetPred or tmodes[0]
        print 'mode',mode
        TX,TY = trainData.getX(mode),trainData.getY(mode)

        umodes = testData.modesToLearn()
        assert mode in umodes,'target predicate %r not in test data' % mode
        UX,UY = testData.getX(mode),testData.getY(mode)

        learner = learn.FixedRateGDLearner(ti.prog,TX,TY,epochs=epochs)

        TP0 = Expt.timeAction(
            'running untrained theory on train data',
            lambda:learner.predict(mode,TX))
        if conf.num_train_predictions_shown>0:
            print 'predictions:'
            d = ti.db.matrixAsSymbolDict(TP0)
            for k in d:
                if k<conf.num_train_predictions_shown:
                    print k,d[k]
        UP0 = Expt.timeAction(
            'running untrained theory on test data',
            lambda:learner.predict(mode,UX))

        Expt.timeAction('training', lambda:learner.train(mode))

        TP1 = Expt.timeAction(
            'running trained theory on train data',
            lambda:learner.predict(mode,TX))
        UP1 = Expt.timeAction(
            'running trained theory on test data',
            lambda:learner.predict(mode,UX))

        Expt.printStats('untrained theory','train',learner,TP0,TY)
        Expt.printStats('..trained theory','train',learner,TP1,TY)
        Expt.printStats('untrained theory','test',learner,UP0,UY)
        Expt.printStats('..trained theory','test',learner,UP1,UY)

        if savedModel:
            Expt.timeAction('saving trained model', lambda:ti.db.serialize(savedModel))

        if savedTestPreds:
            Expt.timeAction('saving test predictions', lambda:Expt.predictionAsProPPRSolutions(savedTestPreds,mode.functor,ti.db,UX,UP1))

        if savedTestExamples:
            Expt.timeAction('saving test examples', lambda:testData.saveProPPRExamples(savedTestExamples,ti.db))

        if savedTrainExamples:
            Expt.timeAction('saving train examples', lambda:trainData.saveProPPRExamples(savedTrainExamples,ti.db))
        if savedTestPreds and savedTestExamples:
            print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' % (savedTestExamples,savedTestPreds)

    @staticmethod
    def predictionAsProPPRSolutions(fileName,theoryPred,db,X,P,append=False):
        """Print X and P in the ProPPR solutions.txt format."""
        fp = open(fileName,'a' if append else 'w')
        dx = db.matrixAsSymbolDict(X)
        dp = db.matrixAsSymbolDict(P)
        for i in range(max(dx.keys())):
            dix = dx[i]
            dip = dp[i]
            assert len(dix.keys())==1,'X row %d is not onehot: %r' % (i,dix)
            x = dix.keys()[0]    
            fp.write('# proved %d\t%s(%s,X1).\t999 msec\n' % (i+1,theoryPred,x))
            scoresdPs = reversed(sorted([(py,y) for (y,py) in dip.items()]))
            for (r,(py,y)) in enumerate(scoresdPs):
                fp.write('%d\t%.18f\t%s(%s,%s).\n' % (r+1,py,theoryPred,x,y))

    @staticmethod
    def timeAction(msg, act):
        """Do an action encoded as a callable function, return the result,
        while printing the elapsed time to stdout."""
        print msg,'...'
        start = time.time()
        result = act()
        print msg,'... done in %.3f sec' % (time.time()-start)
        return result

    @staticmethod
    def printStats(modelMsg,testSet,learner,P,Y):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        print 'eval',modelMsg,'on',testSet,': acc',learner.accuracy(Y,P),'xent',learner.crossEntropy(Y,P)

class BatchExpt(Expt):
    def __init__(self,configDict,options={}):
        super(BatchExpt,self).__init__(configDict)
        self.options=options
    def _run(self,
             initFiles=None, initProgram=None,
             trainData=None, testData=None,
             theoryPred=None,
             savedTestPreds=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None):

        """ Run an experiment, given a whole bunch of parameters.
        trainData, testData: functor -> (X,Y); as from propprExamplesAsData
        theoryPred: if not None, a list of functors to select from *Data for training/testing (assumes i,o)
        savedTestPreds, savedTestExamples, savedTrainExamples: filename; if not None, then
        serialize predictions and examples for later eval with ProPPR tools.
        savedModel: filename; save result of training somewhere
        """
        ti = tensorlog.Interp(initFiles=initFiles,initProgram=initProgram)
        # TODO: should be a parameter, and should work with a sparse parameter vector
        # ti.prog.setWeights(ti.db.vector(declare.ModeDeclaration('rule(o)')))
        
        if theoryPred == None: theoryPred = set(trainData.keys()+testData.keys())
        elif type(theoryPred)==type(""): theoryPred = [theoryPred]
        modes = [declare.ModeDeclaration('%s(i,o)' % p) for p in theoryPred]
        trainModes = [m for m in modes if m.functor in trainData]
        testModes = [m for m in modes if m.functor in testData]

        learner = learn.MultiModeLearner(ti.prog,trainModes,data=trainData,epochs=self.options['epochs'] if 'epochs' in self.options else 5)

        TP0 = Expt.timeAction(
            'running untrained theory on train data',
            lambda:learner.predict(trainModes,data=trainData))
        UP0 = Expt.timeAction(
            'running untrained theory on test data',
            lambda:learner.predict(testModes,data=testData))

        Expt.timeAction('training', lambda:learner.train())

        TP1 = Expt.timeAction(
            'running trained theory on train data',
            lambda:learner.predict(trainModes,data=trainData))
        UP1 = Expt.timeAction(
            'running trained theory on test data',
            lambda:learner.predict(testModes,data=testData))
        
        TY = [trainData[m.functor][1] for m in trainModes]
        UY = [testData[m.functor][1] for m in testModes]
        Expt.printStats('untrained theory','train',learner,TP0,TY)
        Expt.printStats('..trained theory','train',learner,TP1,TY)
        Expt.printStats('untrained theory','test',learner,UP0,UY)
        Expt.printStats('..trained theory','test',learner,UP1,UY)

        if savedModel:
            Expt.timeAction('saving trained model', lambda:ti.db.serialize(savedModel))

        if savedTestPreds:
            open(savedTestPreds,'w').close()
            Expt.timeAction('saving test predictions', lambda:
                [Expt.predictionAsProPPRSolutions(savedTestPreds,m.functor,ti.db,testData[m.functor][0],up1,append=True) for m,up1 in zip(testModes,UP1)])

        if savedTestExamples:
            open(savedTestExamples,'w').close()
            Expt.timeAction('saving test examples', lambda:
                [Expt.dataAsProPPRExamples(savedTestExamples,m.functor,ti.db,testData[m.functor][0],testData[m.functor][1],append=True) for m in testModes])

        if savedTrainExamples:
            open(savedTrainExamples,'w').close()
            Expt.timeAction('saving train examples', lambda:
                [Expt.dataAsProPPRExamples(savedTrainExamples,m.functor,ti.db,trainData[m.functor][0],trainData[m.functor][1],append=True) for m in trainModes])

        if savedTestPreds and savedTestExamples:
            print 'ready for commands like: proppr eval %s %s --metric map' % (savedTestExamples,savedTestPreds)


# a sample main

if __name__=="__main__":

    db = matrixdb.MatrixDB.uncache('tmp-toy.db','test/textcattoy.cfacts')
    trainData = dataset.Dataset.uncacheMatrix('tmp-train.dset',db,'predict/io','train')
    testData = dataset.Dataset.uncacheMatrix('tmp-test.dset',db,'predict/io','test')
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
    prog.setWeights(db.ones())
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'savedModel':'toy-trained.db',
              'savedTestPreds':'tmp-toy-test.solutions.txt',
              'savedTrainExamples':'tmp-toy-train.examples',
              'savedTestExamples':'tmp-toy-test.examples'}
    Expt(params).run()






