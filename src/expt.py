# (C) William W. Cohen and Carnegie Mellon University, 2016

import tensorlog
import declare
import learn
import time

class Expt(object):

    def __init__(self,configDict):
        self.config = configDict

    def run(self):
        self._run(**self.config)

    def _run(self,
             initFiles=None, initProgram=None,
             theoryPred=None, 
             trainMatPair=None, testMatPair=None,
             trainPred=None, testPred=None, 
             savedTestPreds=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None):

        """ Run an experiment, given a whole bunch of parameters.
        theoryPred: functor for mode of the theory predicate to learn (assumed i,o)
        trainMatPair, testMatPair: pairs of matrices (X,Y)
        trainPred, testPred: where to find training/test data, using db.matrixAsTrainingData
        savedTestPreds, savedTestExamples, savedTrainExamples: if not None, then
        serialize predictions and examples for later eval with ProPPR tools.
        savedModel: save result of training somewhere
        """
        mode = declare.ModeDeclaration('%s(i,o)' % theoryPred)
        ti = tensorlog.Interp(initFiles=initFiles,initProgram=initProgram)
        # TODO: should be a parameter, and should work with a sparse parameter vector
        # ti.prog.setWeights(ti.db.vector(declare.ModeDeclaration('rule(o)')))

        if trainMatPair:
            TX,TY = trainMatPair
        else:
            TX,TY = Expt.timeAction(
                'prepare training data',
                lambda:ti.db.matrixAsTrainingData(trainPred,2))

        if testMatPair:
            UX,UY = testMatPair
        else:
            UX,UY = Expt.timeAction(
                'prepare test data',
                lambda:ti.db.matrixAsTrainingData(testPred,2))

        learner = learn.FixedRateGDLearner(ti.prog,TX,TY,epochs=5)

        TP0 = Expt.timeAction(
            'running untrained theory on train data',
            lambda:learner.predict(mode,TX))
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
            Expt.timeAction('saving test predictions', lambda:Expt.predictionAsProPPRSolutions(savedTestPreds,theoryPred,ti.db,UX,UP1))

        if savedTestExamples:
            Expt.timeAction('saving test examples', lambda:Expt.dataAsProPPRExamples(savedTestExamples,theoryPred,ti.db,UX,UY))

        if savedTrainExamples:
            Expt.timeAction('saving train examples', lambda:Expt.dataAsProPPRExamples(savedTrainExamples,theoryPred,ti.db,UX,UY))

        if savedTestPreds and savedTestExamples:
            print 'ready for commands like: proppr eval %s %s --metric map' % (savedTestExamples,savedTestPreds)

    @staticmethod
    def timeAction(msg, act):
        """Do an action encoded as a callable function, return the result,
        while printing the elapsed time to stdout."""
        print msg,'...'
        start = time.time()
        result = act()
        print msg,'... done in in %.3f sec' % (time.time()-start)
        return result

    @staticmethod
    def printStats(modelMsg,testSet,learner,P,Y):
        """Print accuracy and crossEntropy for some named model on a named eval set."""
        print 'eval',modelMsg,'on',testSet,': acc',learner.accuracy(Y,P),'xent',learner.crossEntropy(Y,P)

    @staticmethod
    def dataAsProPPRExamples(fileName,theoryPred,db,X,Y):
        """Convert X and Y to ProPPR examples and store in a file."""
        fp = open(fileName,'w')
        dx = db.matrixAsSymbolDict(X)
        dy = db.matrixAsSymbolDict(Y)
        for i in range(max(dx.keys())):
            dix = dx[i]
            diy = dy[i]
            assert len(dix.keys())==1,'X row %d is not onehot: %r' % (i,dix)
            x = dix.keys()[0]
            fp.write('%s(%s,Y)' % (theoryPred,x))
            for y in diy.keys():
                fp.write('\t+%s(%s,%s)' % (theoryPred,x,y))
            fp.write('\n')

    @staticmethod
    def predictionAsProPPRSolutions(fileName,theoryPred,db,X,P):
        """Print X and P in the ProPPR solutions.txt format."""
        fp = open(fileName,'w')
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


if __name__=="__main__":
    toyparams = {'initFiles':["test/textcattoy.cfacts","test/textcat.ppr"],
                 'theoryPred':'predict',
                 'trainPred':'train',
                 'testPred':'test',
                 'savedModel':'toy-trained.db',
                 'savedTestPreds':'toy-test.solutions.txt',
                 'savedTrainExamples':'toy-train.examples',
                 'savedTestExamples':'toy-test.examples',
    }
    Expt(toyparams).run()

