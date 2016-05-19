# (C) William W. Cohen and Carnegie Mellon University, 2016

import time
import re
import collections

import tensorlog
import declare
import learn
import mutil

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
            Expt.timeAction('saving train examples', lambda:Expt.dataAsProPPRExamples(savedTrainExamples,theoryPred,ti.db,TX,TY))

        if savedTestPreds and savedTestExamples:
            print 'ready for commands like: proppr eval %s %s --metric map' % (savedTestExamples,savedTestPreds)

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

    @staticmethod 
    def propprExamplesAsData(db,fileName):
        """Convert a foo.examples file to a dict of modename->(X,Y)pairs"""
        xsResult = collections.defaultdict(list)
        ysResult = collections.defaultdict(list)
        regex = re.compile('(\w+)\((\w+),(\w+)\)')
        fp = open(fileName)
        for line in fp:
            parts = line.strip().split("\t")
            mx = regex.search(parts[0])
            if mx:
                pred = mx.group(1)
                x = mx.group(2)
                pos = []
                for ans in parts[1:]:
                    #print '=',pred,x,ans
                    label = ans[0]
                    my = regex.search(ans[1:])
                    assert my,'problem at line '+line
                    assert my.group(1)==pred,'mismatched preds at line '+line
                    assert my.group(2)==x,'mismatched x\'s at line '+line
                    if label=='+':
                        pos.append(my.group(3))
                xsResult[pred].append(x)
                ysResult[pred].append(pos)
                #print pred,x,pos
        #print xsResult
        #print ysResult
        result = {}
        for pred in xsResult.keys():
            xRows = map(lambda x:db.onehot(x), xsResult[pred])
            def yRow(ys):
                accum = db.onehot(ys[0])
                for y in ys[1:]:
                    accum = accum + db.onehot(y)
                accum = accum * 1.0/len(ys)
                return accum
            yRows = map(yRow, ysResult[pred])
            result[pred] = (mutil.stack(xRows),mutil.stack(yRows))
        return result

    @staticmethod
    def dataAsProPPRExamples(fileName,theoryPred,db,X,Y,append=False):
        """Convert X and Y to ProPPR examples and store in a file."""
        fp = open(fileName,'a' if append else 'w')
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
        trainData, testData: functor -> (X,Y) as from propprExamplesAsData
        savedTestPreds, savedTestExamples, savedTrainExamples: if not None, then
        serialize predictions and examples for later eval with ProPPR tools.
        savedModel: save result of training somewhere
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
#    Expt(toyparams).run()
    ti = tensorlog.Interp(initFiles=["test/textcattoy.cfacts","test/textcat.ppr"])
    d = Expt.propprExamplesAsData(ti.db,'test/toytrain.examples')
    for pred,(X,Y) in d.items():
        print pred,ti.db.matrixAsSymbolDict(X)
        print pred,ti.db.matrixAsSymbolDict(Y)


