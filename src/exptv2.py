# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for running experiments
#

import time
import logging
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
        return self._run(**self.config)

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
        if targetPred!=None: 
            assert targetPred in tmodes, 'target predicate %r not in training data' % targetPred

        singlePred = targetPred!=None or (targetPred==None and len(tmodes)==1)
        if singlePred:

            mode = targetPred or tmodes[0]
            assert testData.hasMode(mode)
            TX,TY = trainData.getX(mode),trainData.getY(mode)
            UX,UY = testData.getX(mode),testData.getY(mode)
            
            learner = learn.FixedRateGDLearner(ti.prog,epochs=epochs)

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

            Expt.timeAction('training', lambda:learner.train(mode,TX,TY))

            TP1 = Expt.timeAction(
                'running trained theory on train data',
                lambda:learner.predict(mode,TX))
            UP1 = Expt.timeAction(
                'running trained theory on test data',
                lambda:learner.predict(mode,UX))

            Expt.printStats('untrained theory','train',TY,TP0)
            Expt.printStats('..trained theory','train',TY,TP1)
            Expt.printStats('untrained theory','test',UY,UP0)
            testAcc,testXent = Expt.printStats('..trained theory','test',UY,UP1)

        else:
            learner = learn.MultiPredFixedRateGDLearner(ti.prog,epochs=epochs)
            print 'multipred learner',type(learner)

            TP0 = Expt.timeAction(
                'running untrained theory on train data',
                lambda:learner.multiPredict(trainData))
            if conf.num_train_predictions_shown>0:
                logging.warn('sample predictions not implemented')
            UP0 = Expt.timeAction(
                'running untrained theory on test data',
                lambda:learner.multiPredict(testData))

            Expt.timeAction('training', lambda:learner.multiTrain(trainData))

            TP1 = Expt.timeAction(
                'running trained theory on train data',
                lambda:learner.multiPredict(trainData))
            UP1 = Expt.timeAction(
                'running trained theory on test data',
                lambda:learner.multiPredict(testData))

            Expt.printMultiPredStats('untrained theory','train',trainData,TP0)
            Expt.printMultiPredStats('..trained theory','train',trainData,TP1)
            Expt.printMultiPredStats('untrained theory','test',testData,UP0)
            testAcc,testXent = Expt.printMultiPredStats('..trained theory','test',testData,UP1)

        if savedModel:
            Expt.timeAction('saving trained model', lambda:ti.db.serialize(savedModel))

        if singlePred:
            if savedTestPreds:
                Expt.timeAction('saving test predictions for mode %s' % str(mode),
                                lambda:Expt.predictionAsProPPRSolutions(savedTestPreds,mode.functor,ti.db,UX,UP1))

            if savedTestExamples:
                Expt.timeAction('saving test examples for mode %s' % str(mode), 
                                lambda:testData.saveProPPRExamples(savedTestExamples,ti.db,mode=mode))

            if savedTrainExamples:
                Expt.timeAction('saving train examples for mode %s' % str(mode), 
                                lambda:trainData.saveProPPRExamples(savedTrainExamples,ti.db,mode=mode))

        else:
            if savedTestPreds:
                open(savedTestPreds,"w").close() # wipe file first
                def doit():
                    qid=0
                    for mode in testData.modesToLearn():
                        qid+=Expt.predictionAsProPPRSolutions(savedTestPreds,mode.functor,ti.db,UP1.getX(mode),UP1.getY(mode),True,qid) 
                Expt.timeAction('saving test predictions', doit)

            if savedTestExamples:
                Expt.timeAction('saving test examples', 
                                lambda:testData.saveProPPRExamples(savedTestExamples,ti.db))

            if savedTrainExamples:
                Expt.timeAction('saving train examples', 
                                lambda:trainData.saveProPPRExamples(savedTrainExamples,ti.db))
                
        if savedTestPreds and savedTestExamples:
            print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' \
                % (savedTestExamples,savedTestPreds)

        return testAcc,testXent


    @staticmethod
    def predictionAsProPPRSolutions(fileName,theoryPred,db,X,P,append=False,start=0):
        """Print X and P in the ProPPR solutions.txt format."""
        fp = open(fileName,'a' if append else 'w')
        dx = db.matrixAsSymbolDict(X)
        dp = db.matrixAsSymbolDict(P)
        n=max(dx.keys())
        for i in range(n):
            dix = dx[i]
            dip = dp[i]
            assert len(dix.keys())==1,'X %s row %d is not onehot: %r' % (theoryPred,i,dix)
            x = dix.keys()[0]    
            fp.write('# proved %d\t%s(%s,X1).\t999 msec\n' % (i+1+start,theoryPred,x))
            scoresdPs = reversed(sorted([(py,y) for (y,py) in dip.items()]))
            for (r,(py,y)) in enumerate(scoresdPs):
                fp.write('%d\t%.18f\t%s(%s,%s).\n' % (r+1,py,theoryPred,x,y))
        return n

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
    def printStats(modelMsg,testSet,Y,P):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        acc = learn.Learner.accuracy(Y,P)
        xent = learn.Learner.crossEntropy(Y,P)
        print 'eval',modelMsg,'on',testSet,': acc',acc,'xent',xent
        return (acc,xent)

    @staticmethod
    def printMultiPredStats(modelMsg,testSet,goldData,predictedData):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        acc = learn.MultiPredLearner.multiAccuracy(goldData,predictedData)
        xent = learn.MultiPredLearner.multiCrossEntropy(goldData,predictedData)
        print 'eval',modelMsg,'on',testSet,': acc',acc,'xent',xent
        return (acc,xent)

# a sample main

if __name__=="__main__":

    db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
    trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
    testData = dataset.Dataset.uncacheMatrix('tlog-cache/test.dset',db,'predict/io','test')
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
    prog.setWeights(db.ones())
    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'savedModel':'toy-trained.db',
              'savedTestPreds':'tlog-cache/toy-test.solutions.txt',
              'savedTrainExamples':'tlog-cache/toy-train.examples',
              'savedTestExamples':'tlog-cache/toy-test.examples'}
    Expt(params).run()
