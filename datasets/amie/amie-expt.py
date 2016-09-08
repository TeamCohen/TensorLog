import sys

from expt import Expt
import declare
import tensorlog
import learn
import plearn
import funs


def setup(optdict, settings):
    # prog is shortcut to the output optdict, for convenience.
    prog = optdict['prog']

    # the weight vector is sparse - just the constants in the unary predicate rule
    prog.setRuleWeights(prog.db.vector(declare.asMode("rule(i)")))
    
    # set the max recursion depth
    prog.maxDepth = settings['maxDepth']
    
    # be verbose
    # funs.conf.trace = True

    # use a non-default learner, overriding the tracing function,
    # number of epochs, and regularizer
    learner = plearn.ParallelFixedRateGDLearner(
        prog,epochs=settings['epochs'],parallel=settings['para'],regularizer=learn.L2Regularizer())

    #learner = learn.FixedRateGDLearner(
    #    prog,epochs=epochs,regularizer=learn.L2Regularizer())

    #learner = learn.FixedRateSGDLearner(
    #    prog,epochs=epochs,regularizer=learn.L2Regularizer())

    #    learner = plearn.ParallelAdaGradLearner(
    #        prog,epochs=epochs,parallel=40,regularizer=learn.L2Regularizer())
    return learner



if __name__=="__main__":
    #usage: [dataset] [epochs] [maxDepth] [threads]
    
    #get the command-line options for this experiment
    settings = {}
    settings['dataset'] = 'yago2-sample' if len(sys.argv)<=1 else sys.argv[1]
    settings['epochs'] = 30 if len(sys.argv)<=2 else int(sys.argv[2])
    settings['maxDepth'] = 1 if len(sys.argv)<=3 else int(sys.argv[3])
    settings['para'] = 30 if len(sys.argv)<=4 else int(sys.argv[4])


    # first run eval set on untrained model using full KB for inference:

    optdict,args = tensorlog.parseCommandLine([
            '--logging', 'debug', # was: 'warn'
            '--db', 'inputs/{0}-untrained-eval.db|inputs/{0}-db-untrained-eval.cfacts'.format(settings['dataset']),
            '--prog','inputs/{0}.ppr'.format(settings['dataset']), '--proppr',
            '--test', 'inputs/eval.dset|inputs/eval.exam'])
    learner = setup(optdict,settings)
    UP0 = Expt.timeAction(
        'running untrained theory on eval data using full KB',
            lambda:learner.datasetPredict(optdict['testData']))
    Expt.printStats('untrained theory','test',optdict['testData'],UP0)



    # next train the model using a subset of the KB:

    savedModel = 'tmp-cache/%s-trained.db' % settings['dataset']
    optdict,args = tensorlog.parseCommandLine([
        '--logging', 'debug', # was: 'warn'
        '--db', 'inputs/{0}-training.db|inputs/{0}-db-training.cfacts'.format(settings['dataset']),
        '--prog','inputs/{0}.ppr'.format(settings['dataset']), '--proppr',
        '--train','inputs/{0}-train.dset|inputs/{0}-train.exam'.format(settings['dataset']),
        '--test', 'inputs/eval.dset|inputs/eval.exam'])

    learner = setup(optdict,settings)
    TP0 = Expt.timeAction(
        'running untrained theory on train data using training KB',
        lambda:learner.datasetPredict(optdict['trainData']))
    Expt.printStats('untrained theory','train',optdict['trainData'],TP0)    
    Expt.timeAction('training %s' % type(learner).__name__, lambda:learner.train(optdict['trainData']))
    TP1 = Expt.timeAction(
        'running trained theory on train data using training KB',
        lambda:learner.datasetPredict(optdict['trainData']))
    Expt.printStats('..trained theory','train',optdict['trainData'],TP1)
    #Expt.timeAction('saving trained model', lambda:optdict['prog'].db.serialize(savedModel))

    # finally evaluate the trained model, plus those facts excluded for training, on the eval set:

    #optdict,args = tensorlog.parseCommandLine([
    #        '--logging', 'debug', # was: 'warn'
    #        '--db', savedModel,
    #        '--prog','inputs/{0}.ppr'.format(settings['dataset']), '--proppr',
    #        '--test', 'inputs/eval.dset|inputs/eval.exam'])
    
    optdict['db'].addFile('inputs/{0}-foreval.cfacts'.format(settings['dataset']))
    optdict['db'].serialize('tmp-cache/%s-trained-eval.db' % settings['dataset'])
    #learner = setup(optdict,settings)
    UP1 = Expt.timeAction(
        'running trained theory on eval data using full KB',
        lambda:learner.datasetPredict(optdict['testData']))

    testAcc,testXent = Expt.printStats('..trained theory','test',optdict['testData'],UP1)

    savedTestPredictions = 'tmp-cache/%s-eval.solutions.txt' % settings['dataset']
    savedTestExamples = 'tmp-cache/%s-eval.examples' % settings['dataset']
    open(savedTestPredictions,"w").close() # wipe file first
    def doit():
        qid=0
        for mode in optdict['testData'].modesToLearn():
            qid+=Expt.predictionAsProPPRSolutions(savedTestPredictions,mode.functor,optdict['prog'].db,UP1.getX(mode),UP1.getY(mode),True,qid) 
    Expt.timeAction('saving eval predictions', doit)
    Expt.timeAction('saving test examples', 
                    lambda:optdict['testData'].saveProPPRExamples(savedTestExamples,optdict['prog'].db))
    print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' \
                % (savedTestExamples,savedTestPredictions)


    
    
