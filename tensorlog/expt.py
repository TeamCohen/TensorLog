# (C) William W. Cohen and Carnegie Mellon University, 2016

#
# support for running experiments
#

import sys
import time
import logging
import collections
import traceback

from tensorlog import comline
from tensorlog import config
from tensorlog import dataset
from tensorlog import declare
from tensorlog import learn
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import plearn

def fulltype(o):
  return o.__module__ + "." + o.__class__.__name__

class Expt(object):

    def __init__(self,configDict):
        self.config = configDict

    def run(self):
        print 'Expt configuration:',self.config
        return self._run(**self.config)

    def _run(self,
             prog=None, trainData=None, testData=None, targetMode=None,
             savedTestPredictions=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None,
             learner=None):

        """ Run an experiment.

        The stages are
        - if targetMode is specified, extract just the examples from that mode from trainData and testData
        - evaluate the untrained program on the train and test data and print results
        - train on the trainData
        - if savedModel is given, write the learned database, including the trained parameters,
          to that directory.
        - if savedTestPredictions is given, write the test-data predictions in ProPPR format
        - if savedTestExamples (savedTrainExamples) is given, save the training/test examples in ProPPR format
        """

        if targetMode:
            targetMode = declare.asMode(targetMode)
            trainData = trainData.extractMode(targetMode)
            if testData is not None: 
                testData = testData.extractMode(targetMode)

        if not learner: learner = learn.FixedRateGDLearner(prog)

        
        TP0 = Expt.timeAction(
            'running untrained theory on train data',
            lambda:learner.datasetPredict(trainData))
        Expt.printStats('untrained theory','train',trainData,TP0)
        if testData is not None:
          UP0 = Expt.timeAction(
              'running untrained theory on test data',
              lambda:learner.datasetPredict(testData))
          Expt.printStats('untrained theory','test',testData,UP0)

        Expt.timeAction('training %s' % fulltype(learner), lambda:learner.train(trainData))

        TP1 = Expt.timeAction(
            'running trained theory on train data',
            lambda:learner.datasetPredict(trainData))
        if testData is not None:
          UP1 = Expt.timeAction(
              'running trained theory on test data',
              lambda:learner.datasetPredict(testData))

        Expt.printStats('..trained theory','train',trainData,TP1)
        if testData is not None:
          testAcc,testXent = Expt.printStats('..trained theory','test',testData,UP1)
        else:
          testAcc,testXent = None,None

        if savedModel:
            Expt.timeAction('saving trained model', lambda:prog.db.serialize(savedModel))

        if savedTestPredictions and testData:
            #todo move this logic to a dataset subroutine
            open(savedTestPredictions,"w").close() # wipe file first
            def doit():
                qid=0
                for mode in testData.modesToLearn():
                    qid+=Expt.predictionAsProPPRSolutions(savedTestPredictions,mode.functor,prog.db,UP1.getX(mode),UP1.getY(mode),True,qid)
            Expt.timeAction('saving test predictions', doit)

        if savedTestExamples and testData:
            Expt.timeAction('saving test examples',
                            lambda:testData.saveProPPRExamples(savedTestExamples,prog.db))

        if savedTrainExamples:
            Expt.timeAction('saving train examples',
                            lambda:trainData.saveProPPRExamples(savedTrainExamples,prog.db))

        if savedTestPredictions and savedTestExamples and testData:
            print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' \
                % (savedTestExamples,savedTestPredictions)

        return testAcc,testXent


    @staticmethod
    def predictionAsProPPRSolutions(fileName,theoryPred,db,X,P,append=False,start=0):
        """Print X and P in the ProPPR solutions.txt format."""
        fp = open(fileName,'a' if append else 'w')
        dx = db.matrixAsSymbolDict(X,typeName=db.getDomain(theoryPred,2))
        dp = db.matrixAsSymbolDict(P,typeName=db.getRange(theoryPred,2))
        n=max(dx.keys())
        for i in range(n+1):
            #assert i in dp, "keys dp: %s\nkeys dx: %s" % (dp.keys(),dx.keys())
            dix = dx[i]
            dip = {}
            if i in dp:
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
    def printStats(modelMsg,testSet,goldData,predictedData):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        acc = learn.Learner.datasetAccuracy(goldData,predictedData)
        xent = learn.Learner.datasetCrossEntropy(goldData,predictedData,perExample=True)
        print 'eval',modelMsg,'on',testSet,': acc',acc,'xent/ex',xent
        return (acc,xent)

# a useful main

if __name__=="__main__":

    usageLines = [
        'expt-specific options, given after the argument +++:',
        '    --savedModel e      # where e is a filename',
        '    --learner f         # where f is the name of a learner class',
        '    --learnerOpts g     # g is a string that "evals" to a python dict',
        '    --weightEpsilon eps # parameter weights multiplied by eps',
        '    --params p1/k1,..   # comma-sep list of functor/arity pairs'
    ]
    argSpec = ["learner=", "savedModel=", "learnerOpts=", "targetMode=",
               "savedTestPredictions=", "savedTestExamples=", "savedTrainExamples=",
               "params=","weightEpsilon="]
    optdict,args = comline.parseCommandLine(
        sys.argv[1:],
        extraArgConsumer="expt", extraArgSpec=argSpec, extraArgUsage=usageLines
    )

    weightEpsilon = float(optdict.get('--weightEpsilon',1.0))
    print 'weightEpsilon = ',weightEpsilon
    if '--params' in optdict:
        paramSpecs = optdict['--params'].split(",")
        for spec in paramSpecs:
            functor,arity = spec.split("/")
            optdict['db'].markAsParameter(functor,int(arity))
    optdict['prog'].setFeatureWeights(epsilon=weightEpsilon)
    optdict['prog'].setRuleWeights(epsilon=weightEpsilon)
    learner = None
    if 'learner' in optdict:
        try:
            optdict['learner'] = eval(optdict['learner'])
            #so darn hard to get the number of quotes right in Makefile/shell, so eval 'while'...
            while type(optdict['learnerOpts'])==type(""):
                optdict['learnerOpts'] = eval(optdict.get('learnerOpts','{}'))
            print "decoded learner spec to "+repr(optdict['learner'])+" args "+repr(optdict['learnerOpts'])
            learner = optdict['learner'](optdict['prog'], **optdict['learnerOpts'])
        except Exception as ex:
            print 'exception evaluating learner specification "%s"' % optdict['--learner']
            traceback.print_exc(file=sys.stdout)
            raise ex

    params = {'prog':optdict['prog'],
              'trainData':optdict['trainData'],
              'testData':optdict['testData'],
              'learner':learner,
              'savedModel':optdict.get('savedModel'),
              'targetMode':optdict.get('targetMode'),
              'savedTestPredictions':optdict.get('savedTestPredictions'),
              'savedTestExamples':optdict.get('savedTestExamples'),
              'savedTrainExamples':optdict.get('savedTrainExamples'),
    }

    Expt(params).run()
