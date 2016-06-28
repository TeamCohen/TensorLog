# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for running experiments
#

import sys
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

    #TODO targetPred->targetMode
    def _run(self,
             initProgram=None,trainData=None, testData=None, targetPred=None, epochs=5,
             savedTestPreds=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None,
             learnerFactory=None,regularizer=None):

        """ Run an experiment, given a whole bunch of parameters.
        savedTestPreds, savedTestExamples, savedTrainExamples: if not None, then
        serialize predictions and examples for later eval with ProPPR tools.
        savedModel: save result of training somewhere
        """

        ti = tensorlog.Interp(initProgram=initProgram)

        if targetPred: 
            targetPred = declare.asMode(targetPred)
            trainData = trainData.extractMode(targetPred)
            testData = testData.extractMode(targetPred)

        if not learnerFactory:
            learnerFactory = learn.MultiPredFixedRateGDLearner
        learner = learnerFactory(ti.prog,epochs=epochs,regularizer=regularizer)

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
        xent = learn.Learner.crossEntropy(Y,P,perExample=True)
        print 'eval',modelMsg,'on',testSet,': acc',acc,'xent/ex',xent
        return (acc,xent)

    @staticmethod
    def printMultiPredStats(modelMsg,testSet,goldData,predictedData):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        acc = learn.MultiPredLearner.multiAccuracy(goldData,predictedData)
        xent = learn.MultiPredLearner.multiCrossEntropy(goldData,predictedData,perExample=True)
        print 'eval',modelMsg,'on',testSet,': acc',acc,'xent/ex',xent
        return (acc,xent)

# a sample main

if __name__=="__main__":

    if len(sys.argv)<=1 or sys.argv[1]=='textcattoy':
        db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
        trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
        testData = dataset.Dataset.uncacheMatrix('tlog-cache/test.dset',db,'predict/io','test')
        prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
        initWeights = \
            (prog.db.matrixPreimage(declare.asMode("posPair(o,i)")) + \
                 prog.db.matrixPreimage(declare.asMode("negPair(o,i)"))) * 0.5
    elif len(sys.argv)>1 and sys.argv=='matchtoy':
        db = matrixdb.MatrixDB.loadFile('test/matchtoy.cfacts')
        trainData = dataset.Dataset.loadExamples(db,'test/matchtoy-train.exam')
        testData = trainData
        prog = tensorlog.ProPPRProgram.load(["test/matchtoy.ppr"],db=db)
        initWeights = prof.db.ones()
    else:
        assert False,'usage: python exptv2.py [textcattoy|matchtoy]'
        
    prog.setWeights(initWeights)
    def myLearner(prog,**opts):
        return learn.MultiPredFixedRateSGDLearner(prog,miniBatchSize=1,**opts)

    params = {'initProgram':prog,
              'trainData':trainData, 'testData':testData,
              'savedModel':'toy-trained.db',
              'savedTestPreds':'tlog-cache/toy-test.solutions.txt',
              'savedTrainExamples':'tlog-cache/toy-train.examples',
              'savedTestExamples':'tlog-cache/toy-test.examples',
              'learnerFactory':myLearner, 'epochs':5,
              }
    Expt(params).run()
