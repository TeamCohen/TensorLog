# (C) William W. Cohen and Carnegie Mellon University, 2016

import unittest
import logging
import logging.config
import sys
import math

import scipy.sparse

import tensorlog 
import parser
import matrixdb
import bpcompiler
import ops
import learn

# can call a single test with, e.g.,
# python -m unittest testtensorlog.TestSmallProofs.testIf

def maybeNormalize(expectedResultDict):
    if tensorlog.NORMALIZE:
        #softmax normalization
        for k in expectedResultDict:
            expectedResultDict[k] = math.exp(expectedResultDict[k])
        norm = sum(expectedResultDict.values())
        for c in expectedResultDict:
            expectedResultDict[c] /= norm

class TestSmallProofs(unittest.TestCase):
    
    def setUp(self):
        self.db = matrixdb.MatrixDB.loadFile('test/fam.cfacts')
    
    def testIf(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

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
        tensorlog.MAXDEPTH=4
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
        tensorlog.MAXDEPTH=10
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

    # support routines
    # 

    def maybeNormalize(self,expectedResultDict):
        if tensorlog.NORMALIZE:
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
        mode = tensorlog.ModeDeclaration(modeString)
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
        mode = tensorlog.ModeDeclaration(modeString)
        fun = prog.compile(mode)

        y1 = prog.evalSymbols(mode,[inputSymbol]) 
        self.checkDicts(self.db.rowAsSymbolDict(y1), expectedResultDict)

    def only(self,group):
        self.assertEqual(len(group), 1)
        return group[0]

    def checkDicts(self,actual, expected):
        print 'actual:  ',actual
        if expected:
            print 'expected:',expected
            self.assertEqual(len(actual.keys()), len(expected.keys()))
            for k in actual.keys():
                self.assertAlmostEqual(actual[k], expected[k], delta=0.0001)

class TestGrad(unittest.TestCase):

    def setUp(self):
        #TODO write test for this also
        #self.prog = tensorlog.ProPPRProgram.load(["test/testgrad.ppr","test/testgrad.cfacts"])
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


    def gradCheck(self,ruleStrings,modeString,params,xyPairs,expected):
        """
        ruleStrings - a list of tensorlog rules to use with the db.
        modeString - mode for the data.
        params - list of (functor,arity) pairs that gradients will be computed for
        xyPairs - list of pairs (x,[y1,..,yk]) such that the desired result for x is uniform dist over y's
        expected - dict mapping strings encoding facts to expected sign of the gradient
        """
        #build program
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        prog = tensorlog.Program(db=self.db,rules=rules)
        #build dataset
        data = learn.Dataset(self.db)
        for x,ys in xyPairs:
            data.addDataSymbols(modeString,x,ys)
        #mark params: should be pairs (functor,arity)
        prog.db.clearParamMarkings()
        for functor,arity in params:
            prog.db.markAsParam(functor,arity)
        #compute gradient
        learner = learn.Learner(prog,data)
        updates = learner.crossEntropyUpdate(modeString)
        #put the gradient into a single fact-string-indexed dictionary
        updatesWithStringKeys = {}
        for (functor,arity),up in updates.items():
            upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
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

class TestProPPR(unittest.TestCase):

    def setUp(self):
        self.prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
        self.prog.setWeights(self.prog.db.ones())
        self.xsyms,self.X,self.Y = self.loadExamples("test/textcattoy-train.examples",self.prog.db)
        self.numExamples = self.X.get_shape()[0] 
        self.numFeatures = self.X.get_shape()[1] 
        self.mode = tensorlog.ModeDeclaration('predict(i,o)')
        self.numWords = \
            {'dh':4.0, 'ft':5.0, 'rw':3.0, 'sc':5.0, 'bk':5.0, 
             'rb':4.0, 'mv':8.0,  'hs':9.0, 'ji':6.0, 'tf':8.0, 'jm':8.0 }
    
    def testNativeRow(self):
        for i in range(self.numExamples):
            ops.TRACE = False
            pred = self.prog.eval(self.mode,[self.X.getrow(i)])
            d = self.prog.db.rowAsSymbolDict(pred)
            print '= d',d
            if i<4: 
                pass
#                print 'native row',i,self.xsyms[i],d
            if tensorlog.NORMALIZE:
                uniform = {'pos':0.5,'neg':0.5}
                self.checkDicts(d,uniform)
            else:
                self.checkClass(d,self.xsyms[i],'pos',self.numWords)
                self.checkClass(d,self.xsyms[i],'neg',self.numWords)

    def testNativeMatrix(self):
        ops.TRACE = False
        pred = self.prog.eval(self.mode,[self.X])
        d0 = self.prog.db.matrixAsSymbolDict(pred)
        for i,d in d0.items():
#            if i<4: print 'native matrix',i,self.xsyms[i],d
            if tensorlog.NORMALIZE:
                uniform = {'pos':0.5,'neg':0.5}
                self.checkDicts(d,uniform)
            else:
                self.checkClass(d,self.xsyms[i],'pos',self.numWords)
                self.checkClass(d,self.xsyms[i],'neg',self.numWords)

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
        return xsyms,scipy.sparse.vstack(xs),scipy.sparse.vstack(ys)

    def checkDicts(self,actual, expected):
#        print 'actual:  ',actual
        if expected:
#            print 'expected:',expected
            self.assertEqual(len(actual.keys()), len(expected.keys()))
            for k in actual.keys():
                self.assertAlmostEqual(actual[k], expected[k], 0.0001)

if __name__=="__main__":
    ops.TRACE = True
    if len(sys.argv)==1:
        unittest.main()

