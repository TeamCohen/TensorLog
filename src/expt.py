# (C) William W. Cohen and Carnegie Mellon University, 2016

#TODO
# - better learner: 
#    set up, shuffle, and serialize as a database a { mode:(X,Y) } dict
#    enumerate minibatches based on a { mode:(X,Y,offset) dict }
#     - sample non-empty mode based on size
#     - take next n items in the (X,Y) pair and then update the offset
# - generate proppr script (modulo compilation and stuff) ?

import tensorlog
import declare
import learn
import time

def timeAction(msg, act):
    """Do an action encoded as a callable function, return the result,
    while printing the elapsed time to stdout."""
    print msg,'...'
    start = time.time()
    result = act()
    print msg,'... done in in %.3f sec' % (time.time()-start)
    return result

def printStats(modelMsg,testSet,learner,P,Y):
    """Print accuracy and crossEntropy for some named model on a named eval set."""
    print 'eval',modelMsg,'on',testSet,': acc',learner.accuracy(Y,P),'xent',learner.crossEntropy(Y,P)

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

def runExpt(initFiles=None, initProgram=None, theoryPred=None, trainPred=None, testPred=None, 
            savedTestPreds=None, savedTestExamples=None, savedTrainExamples=None, savedModel=None):
    """ Run an experiment, given a whole bunch of parameters.

    theoryPred: functor for mode of the theory predicate to learn (assumed i,o)

    trainPred, testPred: where to find training/test data

    savedTestPreds, savedTestExamples, savedTrainExamples: if not None, then
    serialize predictions and examples for later eval with ProPPR tools.

    savedModel: save result of training somewhere
    """

    mode = declare.ModeDeclaration('%s(i,o)' % theoryPred)
    ti = tensorlog.Interp(initFiles=initFiles)
    # should be a parameter...
    # ti.prog.setWeights(ti.db.vector(declare.ModeDeclaration('rule(o)')))

    TX,TY = timeAction(
        'prepare training data',
        lambda:ti.db.matrixAsTrainingData(trainPred,2))
    learner = learn.FixedRateGDLearner(ti.prog,TX,TY,epochs=5)

    UX,UY = timeAction(
        'prepare test data',
        lambda:ti.db.matrixAsTrainingData(testPred,2))
    learner = learn.FixedRateGDLearner(ti.prog,TX,TY,epochs=5)

    TP0 = timeAction(
        'running untrained theory on train data',
        lambda:learner.predict(mode,TX))

    UP0 = timeAction(
        'running untrained theory on test data',
        lambda:learner.predict(mode,UX))
    
    timeAction('training', lambda:learner.train(mode))

    TP1 = timeAction(
        'running trained theory on train data',
        lambda:learner.predict(mode,TX))

    UP1 = timeAction(
        'running trained theory on test data',
        lambda:learner.predict(mode,UX))
   
    printStats('untrained theory','train',learner,TP0,TY)
    printStats('..trained theory','train',learner,TP1,TY)
    printStats('untrained theory','test',learner,UP0,UY)
    printStats('..trained theory','test',learner,UP1,UY)

    if savedModel:
        timeAction('saving trained model', lambda:ti.db.serialize(savedModel))
    
    if savedTestPreds:
        timeAction('saving test predictions', lambda:predictionAsProPPRSolutions(savedTestPreds,theoryPred,ti.db,UX,UP1))

    if savedTestExamples:
        timeAction('saving test examples', lambda:dataAsProPPRExamples(savedTestExamples,theoryPred,ti.db,UX,UY))

    if savedTrainExamples:
        timeAction('saving train examples', lambda:dataAsProPPRExamples(savedTrainExamples,theoryPred,ti.db,UX,UY))

    if savedTestPreds and savedTestExamples:
        print 'ready for commands like: proppr eval %s %s --metric map' % (savedTestExamples,savedTestPreds)

if __name__=="__main__":
    params = {'initFiles':["wnet.db","wnet-learned.ppr"],
              'theoryPred':'i_hypernym',
              'trainPred':'train_i_hypernym',
              'testPred':'valid_i_hypernym',
              'savedModel':'hypernym-trained.db',
              'savedTestPreds':'hypernym-test.solutions.txt',
              'savedTrainExamples':'hypernym-train.examples',
              'savedTestExamples':'hypernym-test.examples',
    }
    toyparams = {'initFiles':["../../src/test/textcattoy.cfacts","../../src/test/textcat.ppr"],
                 'theoryPred':'predict',
                 'trainPred':'train',
                 'testPred':'test',
                 'savedModel':'toy-trained.db',
                 'savedTestPreds':'toy-test.solutions.txt',
                 'savedTrainExamples':'toy-train.examples',
                 'savedTestExamples':'toy-test.examples',
    }
    runExpt(**params)
