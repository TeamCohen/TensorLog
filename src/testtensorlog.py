# (C) William W. Cohen and Carnegie Mellon University, 2016

# can call a single test with, e.g.,
# python -m unittest testtensorlog.TestSmallProofs.testIf

import unittest
import logging
import logging.config
import collections
import sys
import math
import os
import os.path
import shutil

import funs
import tensorlog 
import declare
import parser
import matrixdb
import bpcompiler
import ops
import funs
import learn
import mutil
import dataset
import exptv2
import exptv1


def maybeNormalize(expectedResultDict):
    #softmax normalization
    for k in expectedResultDict:
        expectedResultDict[k] = math.exp(expectedResultDict[k])
    norm = sum(expectedResultDict.values())
    for c in expectedResultDict:
        expectedResultDict[c] /= norm

class DataBuffer(object):
    def __init__(self,db):
        self.db = db
        self.xSyms = []
        self.ySyms = []
        self.xs = []
        self.ys = []
    def addDataSymbols(self,sx,syList):
        """syList is a list of symbols that are correct answers to input sx
        for the function associated with the given mode."""
        assert len(syList)>0, 'need to have some desired outputs for each input'
        self.xSyms.append(sx)
        self.xs.append(self.db.onehot(sx))
        self.ySyms.append(syList)
        distOverYs = self.db.onehot(syList[0])
        for sy in syList[1:]:
            distOverYs = distOverYs + self.db.onehot(sy)
        distOverYs = distOverYs * (1.0/len(syList))
        self.ys.append(distOverYs)
    def getData(self):
        """Return matrix pair X,Y - inputs and corresponding outputs of the
        function for the given mode."""
        return self.getX(),self.getY()
    def getX(self):
        assert self.xs, 'no data inserted for mode %r in %r' % (mode,self.xs)
        return mutil.stack(self.xs)
    def getY(self):
        assert self.ys, 'no labels inserted for mode %r' % mode
        return mutil.stack(self.ys)



class TestModeDeclaration(unittest.TestCase):

    def testHash(self):
        d = {}
        m1 = declare.ModeDeclaration('foo(i,o)')
        m2 = declare.ModeDeclaration('foo(i, o)')
        self.assertTrue(m1==m2)
        d[m1] = 1.0
        self.assertTrue(m2 in d)

class TestInterp(unittest.TestCase):

    def setUp(self):
        self.ti = tensorlog.Interp('test/textcattoy.cfacts:test/textcat.ppr'.split(':'),proppr=True)

    def testList(self):
        self.ti.list("predict/2")
        self.ti.list("predict/io")
        self.ti.list("hasWord/2")
        self.ti.listAllRules()
        self.ti.listAllFacts()
        print self.ti.eval("predict/io", "pb")

class TestMatrixRecursion(unittest.TestCase):

    def setUp(self):
        self.db = matrixdb.MatrixDB.loadFile('test/fam.cfacts')
    
    def testRec0(self):
        self.matInferenceCheck(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-child(X,Y).'],
                               'r1(i,o)',['william','rachel'],
                               {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})

    def testRec1(self):
        self.matInferenceCheck(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-child(X,Y).'],
                               'p(i,o)',['william','rachel'],
                               {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})

    def testRecBound(self):
        self.matInferenceCheck(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-spouse(X,Y).'],
                               'p(i,o)',['william','rachel'],
                               {0: {'charlie': 1.0/3.0, 'josh': 1.0/3.0, 'susan':1.0/3.0}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})
        self.matInferenceCheck(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-spouse(X,Y).'],
                               'p(i,o)',['william','rachel'],
                               {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}},
                               maxDepth=0)

    def matInferenceCheck(self,ruleStrings,modeString,inputSymbols,expectedResultDict,maxDepth=None):
        print 'testing inference for mode',modeString,'on inputs',inputSymbols,'with rules:'
        for r in ruleStrings:
            print '>',r
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        if maxDepth!=None: prog.maxDepth=maxDepth
        mode = declare.ModeDeclaration(modeString)
        X = mutil.stack(map(lambda s:prog.db.onehot(s), inputSymbols))
        actual = prog.eval(mode,[X])
        print 'compiled functions',prog.function.keys()
        self.checkDictOfDicts(prog.db.matrixAsSymbolDict(actual), expectedResultDict)
        
    def checkDictOfDicts(self,actual,expected):
        print 'actual',actual
        print 'expected',expected
        self.assertTrue(len(actual.keys())==len(expected.keys()))
        for r in actual.keys():
            da = actual[r]
            de = expected[r]
            if not matrixdb.NULL_ENTITY_NAME in de:
                de[matrixdb.NULL_ENTITY_NAME]=0.0
            self.assertTrue(len(da.keys())==len(de.keys()))
            for k in da.keys():
                self.assertAlmostEqual(da[k],de[k],delta=0.05)

class TestSmallProofs(unittest.TestCase):
    
    def setUp(self):
        self.db = matrixdb.MatrixDB.loadFile('test/fam.cfacts')
    
    def testIf(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

    def testFailure(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'lottie', {matrixdb.NULL_ENTITY_NAME:1.0})

    def testRevIf(self):
        self.inferenceCheck(['p(X,Y):-sister(Y,X).'], 'p(i,o)', 'rachel', {'william':1.0})

    def testOr(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).', 'p(X,Y):-sister(X,Y).'], 'p(i,o)', 'william', 
                            {'susan':1.0, 'rachel':1.0, 'lottie':1.0, 'sarah':1.0})

    def testChain(self):
        self.inferenceCheck(['p(X,Z):-spouse(X,Y),sister(Y,Z).'], 'p(i,o)', 'susan', 
                            {'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
        self.inferenceCheck(['p(X,Z):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william', 
                            {'charlotte':1.0, 'lucas':1.0, 'poppy':1.0, 'caroline':1.0, 'elizabeth':1.0})

    def testMid(self):
        self.inferenceCheck(['p(X,Y):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william', 
                            {'sarah': 1.0, 'rachel': 2.0, 'lottie': 2.0})

    def testNest(self):
        self.inferenceCheck(['s(X,Y):-spouse(X,Y).','t(X,Z):-spouse(X,Y),s(Y,Z).'], 't(i,o)', 'susan', {'susan': 1.0}) 

    def testBack1(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y),sister(X,Z).'], 'p(i,o)', 'william', {'susan': 3.0})

    def testBack2(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y),sister(X,Z1),sister(X,Z2).'],'p(i,o)','william',{'susan': 9.0})

    def testRec1(self):
        #TODO check that these work
        tensorlog.DEFAULT_MAXDEPTH=4
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
        tensorlog.DEFAULT_MAXDEPTH=10
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 11.0})

    def testConstOutput(self):
        self.inferenceCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'sarah', {'william': 1.0})
        self.inferenceCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'lottie', {'william': 2.0})

#    def testTrivConstOutput(self):
#        self.inferenceCheck(['sis(X,W):-assign(W,william).'], 'sis(i,o)', 'sarah', {'william': 1.0})
#        self.inferenceCheck(['sis(X,W):-assign(W,william).'], 'sis(i,o)', 'lottie', {'william': 1.0})

    def testConstChain1(self):
        self.inferenceCheck(['p(X,S) :- assign(S,susan),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

    def testConstChain2(self):
        #self.inferenceCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','rachel',{'pos':0.0})
        self.inferenceCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','sarah',{'pos':1.0})
        self.inferenceCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','lottie',{'pos':2.0})

    def testAltChain(self):
        self.inferenceCheck(['p(X,W) :- spouse(X,W),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

    def testProppr1(self):
        w = 7*self.db.onehot('r1')+3*self.db.onehot('r2')        
        self.propprInferenceCheck(w,['p(X,Y):-sister(X,Y) {r1}.','p(X,Y):-spouse(X,Y) {r2}.'],'p(i,o)',
                                  'william', {'sarah': 7.0, 'rachel': 7.0, 'lottie': 7.0, 'susan': 3.0})

    def testProppr2(self):
        w = 3*self.db.onehot('r2')
        self.propprInferenceCheck(w,['p(X,Y):-spouse(Y,X) {r2}.'],'p(i,o)',
                                  'susan', {'william': 3.0})

    # support routines
    # 

    def maybeNormalize(self,expectedResultDict):
        norm = sum(expectedResultDict.values())
        for c in expectedResultDict:
            expectedResultDict[c] /= norm


    def inferenceCheck(self,ruleStrings,modeString,inputSymbol,expectedResultDict):
        print 'testing inference for mode',modeString,'on input',inputSymbol,'with rules:'
        maybeNormalize(expectedResultDict)
        for r in ruleStrings:
            print '>',r
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        mode = declare.ModeDeclaration(modeString)
        fun = prog.compile(mode)
        y1 = prog.evalSymbols(mode,[inputSymbol]) 
        self.checkDicts(self.db.rowAsSymbolDict(y1), expectedResultDict)


    def propprInferenceCheck(self,weightVec,ruleStrings,modeString,inputSymbol,expectedResultDict):
        print 'testing inference for mode',modeString,'on input',inputSymbol,'with proppr rules:'
        maybeNormalize(expectedResultDict)
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.ProPPRProgram(db=self.db,rules=rules,weights=weightVec)
        mode = declare.ModeDeclaration(modeString)
        fun = prog.compile(mode)

        y1 = prog.evalSymbols(mode,[inputSymbol]) 
        self.checkDicts(self.db.rowAsSymbolDict(y1), expectedResultDict)

    def only(self,group):
        self.assertEqual(len(group), 1)
        return group[0]

    def checkDicts(self,actual, expected):
        print 'actual:  ',actual
        if not matrixdb.NULL_ENTITY_NAME in expected:
            expected[matrixdb.NULL_ENTITY_NAME]=0.0
        if expected:
            print 'expected:',expected
            self.assertEqual(len(actual.keys()), len(expected.keys()))
            for k in actual.keys():
                self.assertAlmostEqual(actual[k], expected[k], delta=0.05)


class TestMultiRowOps(unittest.TestCase):
    def setUp(self):
        #logging.basicConfig(level=logging.DEBUG)
        self.db = matrixdb.MatrixDB.loadFile('test/fam.cfacts')
    def testThing(self):
        self.compareCheck([
        "p(X,Y):-q(X,Y){r1}.",
        "p(X,Y):-r(X,Y){r2}.",
        "p(X,Y):-spouse(X,Y){r3}.",
        "q(X,Y):-child(X,Y){rq}.", # DefinedPredOp should return multi-row results
        "r(X,Y):-s(X,Y){rr}.",
        "s(X,Y):-r(X,Y){rs}.", # NullFunction should return multi-row results
        ],"p(i,o)",["william","rachel"],[
            {#'rachel':1.0,'sarah':1.0,'lottie':1.0,
            'susan':1.0,
            'josh':1.0,'charlie':1.0},
            {'caroline':1.0,'elizabeth':1.0}])
    def compareCheck(self,ruleStrings,modeString,inputSymbols,expectedResultDicts):
        for d in expectedResultDicts:
            maybeNormalize(d)
            d[matrixdb.NULL_ENTITY_NAME] = 0
        self.inferenceCheck(ruleStrings,modeString,inputSymbols,expectedResultDicts)
        self.predictCheck(ruleStrings,modeString,inputSymbols,expectedResultDicts)
    def inferenceCheck(self,ruleStrings,modeString,inputSymbols,expectedResultDicts):
        print '\n\ntesting inference for mode',modeString,'on input',inputSymbols,'with rules:'
        for r in ruleStrings:
            print '>',r
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        mode = declare.ModeDeclaration(modeString)
        
        #trainingData = self.db.createPartner()
        #trainingData.addFile(trainingDataFile)
        #trainSpec = (mode.functor,mode.arity)
        #X,Y = trainingData.matrixAsTrainingData(*trainSpec)
        #self.learner = learn.FixedRateGDLearner(self.prog,X,Y,epochs=5)
        #P0 = self.learner.predict(mode,X)
        
        fun = prog.compile(mode)
        for i in range(len(inputSymbols)):
            y1 = prog.evalSymbols(mode,[inputSymbols[i]])
            self.checkDicts(self.db.rowAsSymbolDict(y1), expectedResultDicts[i])
    def predictCheck(self,ruleStrings,modeString,inputSymbols,expectedResultDicts):
        print '\n\ntesting predictions for mode',modeString,'on input',inputSymbols,'with rules:'
        for r in ruleStrings:
            print '>',r
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        mode = declare.ModeDeclaration(modeString)
        
        td = []
        print 'training data:'
        for i in range(len(inputSymbols)):
            sol = expectedResultDicts[i].keys()[0]
            td.append("\t".join([mode.functor,inputSymbols[i],sol]))
            print td[-1]
        trainingData = self.db.createPartner()
        trainingData.addLines(td)
        trainSpec = (mode.functor,mode.arity)
        X,Y = trainingData.matrixAsTrainingData(*trainSpec)
        learner = learn.FixedRateGDLearner(prog,X,Y,epochs=5)
        P0 = learner.predict(mode,X)
        
    def checkDicts(self,actual, expected):
        print 'actual:  ',actual
        if expected:
            print 'expected:',expected
            self.assertEqual(len(actual.keys()), len(expected.keys()))
            for k in actual.keys():
                self.assertAlmostEqual(actual[k], expected[k], delta=0.0001)

class TestGrad(unittest.TestCase):

    def setUp(self):
        self.db = matrixdb.MatrixDB.loadFile('test/fam.cfacts')
    
    def testIf(self):
        rules = ['p(X,Y):-sister(X,Y).']
        mode = 'p(i,o)'  
        params = [('sister',2)] 
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','sarah'])], 
                       {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
        self.gradCheck(rules, mode, params, 
                       [('william',['lottie'])], 
                       {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

    def testRevIf(self):
        rules = ['p(X,Y):-parent(Y,X).']
        mode = 'p(i,o)'  
        params = [('parent',2)] 
        self.gradCheck(rules, mode, params,
                       [('lottie',['charlotte'])], 
                       {'parent(charlotte,lottie)': +1,'parent(lucas,lottie)': -1})

    def testChain1(self):
        rules = ['p(X,Z):-sister(X,Y),child(Y,Z).']
        mode = 'p(i,o)'  
        self.gradCheck(rules,mode,
                       [('sister',2)], 
                       [('william',['caroline','elizabeth'])],
                       {'sister(william,rachel)': +1,'sister(william,lottie)': -1})
        self.gradCheck(rules,mode,
                       [('child',2)], 
                       [('william',['caroline','elizabeth'])],
                       {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1})

        self.gradCheck(rules,mode,
                       [('child',2),('sister',2)], 
                       [('william',['caroline','elizabeth'])],
                       {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1, 'sister(william,rachel)': +1,'sister(william,lottie)': -1})

    def testChain2(self):
        rules = ['p(X,Z):-spouse(X,Y),sister(Y,Z).']
        mode = 'p(i,o)'  
        self.gradCheck(rules,mode,
                       [('sister',2)], 
                       [('susan',['rachel'])],
                       {'sister(william,rachel)': +1,'sister(william,lottie)': -1})


    def testPrintf(self):
        rules = ['p(X,Z1):-printf(X,X1),spouse(X1,Y),printf(Y,Y1),sister(Y1,Z),printf(Z,Z1).']
        mode = 'p(i,o)'  
        self.gradCheck(rules,mode,
                       [('sister',2)], 
                       [('susan',['rachel'])],
                       {'sister(william,rachel)': +1,'sister(william,lottie)': -1})

    def testCall1(self):
        rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-q(Z,W).']
        mode = 'p(i,o)'  
        params = [('sister',2)] 
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','sarah'])], 
                       {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
        self.gradCheck(rules, mode, params, 
                       [('william',['lottie'])], 
                       {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

    def testCall2(self):
        rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-r(Z,W).','r(Z,W):-q(Z,W).']
        mode = 'p(i,o)'  
        params = [('sister',2)] 
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','sarah'])], 
                       {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
        self.gradCheck(rules, mode, params, 
                       [('william',['lottie'])], 
                       {'sister(william,rachel)': -1,'sister(william,lottie)': +1})


    def testSplit(self):
        rules = ['p(X,Y):-sister(X,Y),child(Y,Z),young(Z).']
        mode = 'p(i,o)'
        params = [('child',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['lottie'])],
                       {'child(lottie,lucas)': +1,'child(lottie,charlotte)': +1,'child(sarah,poppy)': -1})
        params = [('sister',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['lottie'])],
                       {'sister(william,lottie)': +1,'sister(william,sarah)': -1})

    def testOr(self):
        rules = ['p(X,Y):-child(X,Y).', 'p(X,Y):-sister(X,Y).']
        mode = 'p(i,o)'
        params = [('sister',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['charlie','rachel'])],
                       {'sister(william,rachel)': +1,'sister(william,sarah)': -1,'sister(william,lottie)': -1})
        params = [('child',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['charlie','rachel'])],
                       {'child(william,charlie)': +1,'child(william,josh)': -1})
        params = [('child',2),('sister',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['charlie','rachel'])],
                       {'child(william,charlie)': +1,'child(william,josh)': -1,'sister(william,rachel)': +1,'sister(william,sarah)': -1})


    def testWeightedVec(self):
        rules = ['p(X,Y):-sister(X,Y),assign(R,r1),feat(R).','p(X,Y):-child(X,Y),assign(R,r2),feat(R).']
        mode = 'p(i,o)'
        params = [('sister',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','charlie'])],
                       {'sister(william,rachel)': +1,'sister(william,sarah)': -1})
        params = [('child',2)]
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','charlie'])],
                       {'child(william,charlie)': +1,'child(william,josh)': -1})
        params = [('feat',1)]
        self.gradCheck(rules, mode, params,
                       [('william',['josh','charlie'])],
                       {'feat(r1)': -1,'feat(r2)': +1})
        self.gradCheck(rules, mode, params,
                       [('william',['rachel','sarah','lottie'])],
                       {'feat(r1)': +1,'feat(r2)': -1})

    def gradCheck(self,ruleStrings,modeString,params,xyPairs,expected):
        """
        expected - dict mapping strings encoding facts to expected sign of the gradient
        """
        mode = declare.ModeDeclaration(modeString)
        (prog,updates) = self.gradUpdates(ruleStrings,mode,params,xyPairs)
        #put the gradient into a single fact-string-indexed dictionary
        updatesWithStringKeys = {}
        for (functor,arity),up in updates.items():
            #print 'up for',functor,arity,'is',up
            upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
            #print 'upDict',upDict,'updates keys',updates.keys()
            for fact,gradOfFact in upDict.items():
                updatesWithStringKeys[str(fact)] = gradOfFact
        self.checkDirections(updatesWithStringKeys,expected)

    def checkDirections(self,actualGrad,expectedDir):
        #TODO allow expected to contain zeros?
        for fact,sign in expectedDir.items():
            print fact,'expected sign',sign,'grad',actualGrad.get(fact)
            if not fact in actualGrad: print 'actualGrad',actualGrad
            self.assertTrue(fact in actualGrad)
            self.assertTrue(actualGrad[fact] * sign > 0)

    def gradUpdates(self,ruleStrings,mode,params,xyPairs):
        """
        ruleStrings - a list of tensorlog rules to use with the db.
        modeString - mode for the data.
        params - list of (functor,arity) pairs that gradients will be computed for
        xyPairs - list of pairs (x,[y1,..,yk]) such that the desired result for x is uniform dist over y's

        return (program,updates)
        """
        #build program
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        #build dataset
        data = DataBuffer(self.db)
        for x,ys in xyPairs:
            data.addDataSymbols(x,ys)
        #mark params: should be pairs (functor,arity)
        prog.db.clearParamMarkings()
        for functor,arity in params:
            prog.db.markAsParam(functor,arity)
        #compute gradient
        learner = learn.Learner(prog)
        updates = learner.crossEntropyGrad(mode,data.getX(),data.getY())
        return prog,updates
    
class TestProPPR(unittest.TestCase):

    def setUp(self):
        #logging.basicConfig(level=logging.DEBUG)
        self.prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
        self.labeledData = self.prog.db.createPartner()
        self.prog.db.moveToPartner(self.labeledData,'train',2)
        self.prog.db.moveToPartner(self.labeledData,'test',2)
        self.prog.setWeights(self.prog.db.ones())
        self.xsyms,self.X,self.Y = self.loadExamples("test/textcattoy-train.examples",self.prog.db)
        self.numExamples = self.X.get_shape()[0] 
        self.numFeatures = self.X.get_shape()[1] 
        self.mode = declare.ModeDeclaration('predict(i,o)')
        self.numWords = \
            {'dh':4.0, 'ft':5.0, 'rw':3.0, 'sc':5.0, 'bk':5.0, 
             'rb':4.0, 'mv':8.0,  'hs':9.0, 'ji':6.0, 'tf':8.0, 'jm':8.0 }
        self.rawPos = "dh ft rw sc bk rb".split()
        self.rawNeg = "mv hs ji tf jm".split()
        self.rawData = {'dh':	'a	pricy	doll	house',
                        'ft':	'a	little	red	fire	truck',
                        'rw':	'a	red	wagon',
                        'sc':	'a	pricy	red	sports	car',
                        'bk':	'punk	queen	barbie	and	ken',
                        'rb':	'a	little	red	bike',
                        'mv':	'a	big	7-seater	minivan	with	an	automatic	transmission',
                        'hs':	'a	big	house	in	the	suburbs	with	crushing	mortgage',
                        'ji':	'a	job	for	life	at	IBM',
                        'tf':	'a	huge	pile	of	tax	forms	due	yesterday',
                        'jm':	'huge	pile	of	junk	mail	bills	and	catalogs'}
    
    def testDBKeys(self):
        self.assertTrue(self.prog.db.stab.hasId(matrixdb.NULL_ENTITY_NAME))

    def testNativeRow(self):
        for i in range(self.numExamples):
            pred = self.prog.eval(self.mode,[self.X.getrow(i)])
            d = self.prog.db.rowAsSymbolDict(pred)
            if i<4: 
                pass
            uniform = {'pos':0.5,'neg':0.5}
            self.checkDicts(d,uniform)

    def testNativeMatrix(self):
        pred = self.prog.eval(self.mode,[self.X])
        d0 = self.prog.db.matrixAsSymbolDict(pred)
        for i,d in d0.items():
            uniform = {'pos':0.5,'neg':0.5,}
            self.checkDicts(d,uniform)

    #seg fault on mac
    def testGradMatrix(self):
        data = DataBuffer(self.prog.db)
        X,Y = self.labeledData.matrixAsTrainingData('train',2)
        learner = learn.Learner(self.prog)
        updates =  learner.crossEntropyGrad(declare.ModeDeclaration('predict(i,o)'),X,Y)
        w = updates[('weighted',1)]
        def checkGrad(i,x,psign,nsign):
            ri = w.getrow(i)            
            di = self.prog.db.rowAsSymbolDict(ri)
            for toki in self.rawData[x].split("\t"):
                posToki = toki+'_pos'
                negToki = toki+'_neg'
                self.assertTrue(posToki in di)
                self.assertTrue(negToki in di)
                self.assertTrue(di[posToki]*psign > 0)
                self.assertTrue(di[negToki]*nsign > 0)
        for i,x in enumerate(self.rawPos):
            checkGrad(i,x,+1,-1)
        for i,x in enumerate(self.rawNeg):
            checkGrad(i+len(self.rawPos),x,-1,+1)

    def testLabeledData(self):
        self.assertTrue(self.labeledData.inDB('train',2))
        self.assertTrue(self.labeledData.inDB('test',2))
        self.assertFalse(self.prog.db.inDB('train',2))
        self.assertFalse(self.prog.db.inDB('test',2))

    def abandoned(self):
        learner = learn.MultiModeLearner(self.prog,[mode],data={mode.functor:(X,Y)},epochs=5)
        P0 = learner.predict([mode],Xs=[X])[0]
        acc0 = learner.accuracy(Y,P0,stack=False)
        xent0 = learner.crossEntropy(Y,P0,stack=False)

        learner.train()
        P1 = learner.predict([mode],Xs=[X])[0]
        acc1 = learner.accuracy(Y,P1,stack=False)
        xent1 = learner.crossEntropy(Y,P1,stack=False)
        
        self.assertTrue(acc0<acc1)
        self.assertTrue(xent0>xent1)
        self.assertTrue(acc1==1)
        print 'toy train: acc1',acc1,'xent1',xent1

        TX,TY = self.labeledData.matrixAsTrainingData('test',2)
        P2 = learner.predict([mode],Xs=[TX])[0]
        acc2 = learner.accuracy(TY,P2,stack=False)
        xent2 = learner.crossEntropy(TY,P2,stack=False)
        print 'toy test: acc2',acc2,'xent2',xent2
        self.assertTrue(acc2==1)

    def testMultiLearn(self):
        mode = declare.ModeDeclaration('predict(i,o)')
        
        dset = dataset.Dataset.loadExamples(self.prog.db,"test/textcattoy-train.examples",proppr=True)
        
        for mode in dset.modesToLearn():
            X = dset.getX(mode)
            Y = dset.getY(mode)
            print mode
            print "\t"+mutil.summary(X)
            print "\t"+mutil.summary(Y)
        #X,Y = self.labeledData.matrixAsTrainingData('train',2)
        #print mutil.summary(X)
        #print mutil.summary(Y)
        
        learner = learn.MultiPredFixedRateGDLearner(self.prog,epochs=5)
        P0 = learner.multiPredict(dset)
        acc0 = learner.multiAccuracy(dset,P0)
        xent0 = learner.multiCrossEntropy(dset,P0)
        print 'toy train: acc0',acc0,'xent1',xent0

        learner.multiTrain(dset)
        
        P1 = learner.multiPredict(dset)
        acc1 = learner.multiAccuracy(dset,P1)
        xent1 = learner.multiCrossEntropy(dset,P1)
        print 'toy train: acc1',acc1,'xent1',xent1
        
        self.assertTrue(acc0<acc1)
        self.assertTrue(xent0>xent1)
        self.assertTrue(acc1==1)

        Udset = dataset.Dataset.loadExamples(self.prog.db,"test/textcattoy-test.examples",proppr=True)
        
        P2 = learner.multiPredict(Udset)
        acc2 = learner.multiAccuracy(Udset,P2)
        xent2 = learner.multiCrossEntropy(Udset,P2)
        print 'toy test: acc2',acc2,'xent2',xent2
        
        self.assertTrue(acc2==1)
        ##

        
    #seg fault on mac
    def testLearn(self):
        mode = declare.ModeDeclaration('predict(i,o)')
        X,Y = self.labeledData.matrixAsTrainingData('train',2)
        learner = learn.FixedRateGDLearner(self.prog,epochs=5)
        P0 = learner.predict(mode,X)
        acc0 = learner.accuracy(Y,P0)
        xent0 = learner.crossEntropy(Y,P0)

        learner.train(mode,X,Y)
        P1 = learner.predict(mode,X)
        acc1 = learner.accuracy(Y,P1)
        xent1 = learner.crossEntropy(Y,P1)
        
        self.assertTrue(acc0<acc1)
        self.assertTrue(xent0>xent1)
        self.assertTrue(acc1==1)
        print 'toy train: acc1',acc1,'xent1',xent1

        TX,TY = self.labeledData.matrixAsTrainingData('test',2)
        P2 = learner.predict(mode,TX)
        acc2 = learner.accuracy(TY,P2)
        xent2 = learner.crossEntropy(TY,P2)
        print 'toy test: acc2',acc2,'xent2',xent2
        self.assertTrue(acc2==1)

    def checkClass(self,d,sym,lab,expected):
        self.assertEqual(d[lab], expected[sym])

    def loadExamples(self,filename,db):
        xsyms = []
        xs = []
        ys = []
        for line in open(filename):
            sx,sy = line.strip().split("\t")
            xsyms.append(sx)
            xs.append(db.onehot(sx))
            ys.append(db.onehot(sy))
        return xsyms,mutil.stack(xs),mutil.stack(ys)

    def checkDicts(self,actual, expected):
        print 'actual:  ',actual
        if not matrixdb.NULL_ENTITY_NAME in expected:
            expected[matrixdb.NULL_ENTITY_NAME]=0.0
        if expected:
            print 'expected:',expected
            self.assertEqual(len(actual.keys()), len(expected.keys()))
            for k in actual.keys():
                self.assertAlmostEqual(actual[k], expected[k], delta=0.05)

class TestExpt(unittest.TestCase):

    def setUp(self):
        if not os.path.exists("tlog-cache"):
            os.mkdir("tlog-cache")
        for f in os.listdir("tlog-cache"):
            p = os.path.join("tlog-cache",f)
            if os.path.isdir(p):
                print 'removing dir',p
                shutil.rmtree(p)
            else:
                print 'removing file',p
                os.remove(p)


    def testExptV1(self):
        params = {'initFiles':["test/textcattoy.cfacts","test/textcat.ppr"],
                  'theoryPred':'predict',
                  'trainPred':'train',
                  'testPred':'test',
                  'savedModel':'tlog-cache/toy-trained.db',
                  'savedTestPreds':'tlog-cache/toy-test.solutions.txt',
                  'savedTrainExamples':'tlog-cache/toy-train.examples',
                  'savedTestExamples':'tlog-cache/toy-test.examples',
              }
        exptv1.Expt(params).run()
        

    def testMToyExpt(self):
        acc2,xent1 = self.runMToyExpt()

    def testTCToyExpt(self):
        #test serialization and uncaching by running the experiment 2x
        acc1,xent1 = self.runTCToyExpt()
        acc2,xent2 = self.runTCToyExpt()
        print 'acc:',acc1,'/',acc2,'xent',xent1,'/',xent2
        self.assertAlmostEqual(acc1,acc2)
        self.assertAlmostEqual(acc1,1.0)
        self.assertAlmostEqual(xent1,xent2)

    def runTCToyExpt(self):
        db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
        db.listing()
        trainData = dataset.Dataset.uncacheMatrix('tlog-cache/train.dset',db,'predict/io','train')
        testData = dataset.Dataset.uncacheMatrix('tlog-cache/test.dset',db,'predict/io','test')
        print 'trainData:\n','\n'.join(trainData.pprint())
        print 'testData"\n','\n'.join(testData.pprint())
        prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr"],db=db)
        prog.setWeights(db.ones())
        params = {'initProgram':prog,
                  'trainData':trainData, 'testData':testData,
                  'savedModel':'toy-trained.db',
                  'savedTestPreds':'tlog-cache/toy-test.solutions.txt',
                  'savedTrainExamples':'tlog-cache/toy-train.examples',
                  'savedTestExamples':'tlog-cache/toy-test.examples'}
        return exptv2.Expt(params).run()

    def runMToyExpt(self):
        db = matrixdb.MatrixDB.uncache('tlog-cache/matchtoy.db','test/matchtoy.cfacts')
        db.listing()
        trainData = dataset.Dataset.uncacheExamples('tlog-cache/mtoy-train.dset',db,'test/matchtoy-train.exam',proppr=False)
        testData = trainData
        print 'trainData:\n','\n'.join(trainData.pprint())
        prog = tensorlog.ProPPRProgram.load(["test/matchtoy.ppr"],db=db)
        prog.setWeights(db.ones())
        params = {'initProgram':prog,'trainData':trainData, 'testData':testData}
        return exptv2.Expt(params).run()

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.db = matrixdb.MatrixDB.loadFile('test/matchtoy.cfacts')

    def testProPPRLoadExamples(self):
        self.checkMatchExamples('test/matchtoy-train.examples', proppr=True)

    def testLoadExamples(self):
        self.checkMatchExamples('test/matchtoy-train.exam', proppr=False)

    def checkMatchExamples(self,filename,proppr):
        dset = dataset.Dataset.loadExamples(self.db,filename,proppr=proppr)
        modes = dset.modesToLearn()
        self.assertEqual(len(modes),2)
        m1 = declare.asMode("match/io")
        m2 = declare.asMode("amatch/io")
        self.assertTrue(m1 in modes); self.assertTrue(m2 in modes)
        x1 = dset.getX(m1); y1 = dset.getY(m1)
        x2 = dset.getX(m2); y2 = dset.getY(m2)
        ax1,ay1,ax2,ay2 = map(lambda m:self.db.matrixAsSymbolDict(m), (x1,y1,x2,y2))
        ex1 = {0: {'r1': 1.0}, 1: {'r3': 1.0}}
        ey1 = {0: {'r1': 0.5, 'r2': 0.5}, 1: {'r4': 0.5, 'r3': 0.5}}
        ex2 = {0: {'a2': 1.0}, 1: {'a4': 1.0}}
        ey2 = {0: {'a1': 0.5, 'a2': 0.5}, 1: {'a3': 0.5, 'a4': 0.5}}
        for actual,expected in [(ax1,ex1),(ax2,ex2),(ay1,ey1),(ay2,ey2)]:
            for i in (0,1):
                self.checkDicts(actual,expected)
        
    def checkDicts(self,actual,expected):
        #print 'actual:  ',actual
        #print 'expected:',expected
        self.assertEqual(len(actual.keys()), len(expected.keys()))
        for k in actual.keys():
            self.assertEqual(actual[k], expected[k])

if __name__=="__main__":
    if len(sys.argv)==1:
        unittest.main()

