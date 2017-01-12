import theanoxcomp
import tensorlog
import declare
import testtensorlog
import matrixdb
import parser
import mutil

import unittest
import sys
import theano

TESTED_COMPILERS = [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler,
]

class TestXCSmallProofs(testtensorlog.TestSmallProofs):

    def testIf(self):
        self.xcompCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

    def testFailure(self):
        self.xcompCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'lottie', {matrixdb.NULL_ENTITY_NAME:1.0})

    def testRevIf(self):
        self.xcompCheck(['p(X,Y):-sister(Y,X).'], 'p(i,o)', 'rachel', {'william':1.0})

    def testOr(self):
        self.xcompCheck(['p(X,Y):-spouse(X,Y).', 'p(X,Y):-sister(X,Y).'], 'p(i,o)', 'william',
                        {'susan':1.0, 'rachel':1.0, 'lottie':1.0, 'sarah':1.0})

    def testChain(self):
        self.xcompCheck(['p(X,Z):-spouse(X,Y),sister(Y,Z).'], 'p(i,o)', 'susan',
                        {'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
        self.xcompCheck(['p(X,Z):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
                        {'charlotte':1.0, 'lucas':1.0, 'poppy':1.0, 'caroline':1.0, 'elizabeth':1.0})


    def testMid(self):
        self.xcompCheck(['p(X,Y):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
                        {'sarah': 1.0, 'rachel': 2.0, 'lottie': 2.0})

    def testNest(self):
        self.xcompCheck(['s(X,Y):-spouse(X,Y).','t(X,Z):-spouse(X,Y),s(Y,Z).'], 't(i,o)', 'susan', {'susan': 1.0})

    def testBack1(self):
        self.xcompCheck(['p(X,Y):-spouse(X,Y),sister(X,Z).'], 'p(i,o)', 'william', {'susan': 3.0})

    def testBack2(self):
        self.xcompCheck(['p(X,Y):-spouse(X,Y),sister(X,Z1),sister(X,Z2).'],'p(i,o)','william',{'susan': 9.0})

    def testRec1(self):
        tensorlog.DEFAULT_MAXDEPTH=4
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
        tensorlog.DEFAULT_MAXDEPTH=10
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 11.0})

    def testConstOutput(self):
        self.xcompCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'sarah', {'william': 1.0})
        self.xcompCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'lottie', {'william': 2.0})

    def testConstChain1(self):
        self.xcompCheck(['p(X,S) :- assign(S,susan),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

    def testConstChain2(self):
        self.xcompCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','sarah',{'pos':1.0})
        self.xcompCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','lottie',{'pos':2.0})

    def testAltChain(self):
        # I believe the problem is it's hard to mix sparse/dense, or that somehow a dense dot.0
        # is being mixed in with the dense outputs...
#         funElemwise{Add}[(0, 0)] [id A] ''
#         |Softmax [id B] ''
#         | |SparseDot [id C] ''
#         |   |Elemwise{Mul}[(0, 1)] [id D] ''
#         |   | |n0__X [id E]
#         |   | |SparseDot [id F] ''
#         |   |   |InplaceDimShuffle{1,0} [id G] ''
#         |   |   | |SparseDot [id H] ''
#         |   |   |   |M__child_io [id I]
#         |   |   |   |InplaceDimShuffle{1,0} [id J] '__ones.T'
#         |   |   |     |__ones [id K]
#         |   |   |SparseTranspose [id L] ''
#         |   |     |M__sister_io [id M]
#         |   |M__spouse_io [id N]
#         |nullSmoothing [id O]
      self.xcompCheck(['p(X,W) :- spouse(X,W),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})
      pass

    def testProppr1(self):
        w = 7*self.db.onehot('r1')+3*self.db.onehot('r2')
        self.propprXCompCheck(w,['p(X,Y):-sister(X,Y) {r1}.','p(X,Y):-spouse(X,Y) {r2}.'],'p(i,o)',
                              'william', {'sarah': 7.0, 'rachel': 7.0, 'lottie': 7.0, 'susan': 3.0})

    def testProppr2(self):
        w = 3*self.db.onehot('r2')
        self.propprXCompCheck(w,['p(X,Y):-spouse(Y,X) {r2}.'],'p(i,o)',
                              'susan', {'william': 3.0})

    def testReuse1(self):
        self.xcompCheck(['p(X,Y) :- r(X,Z),r(Z,Y).', 'r(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william',
                        {'william':1.0})


    def xcompCheck(self,ruleStrings,modeString,inputSymbol,expectedResultDict):
        self._xcompCheck('vanilla',None,ruleStrings,modeString,inputSymbol,expectedResultDict)

    def propprXCompCheck(self,weightVec,ruleStrings,modeString,inputSymbol,expectedResultDict):
        self._xcompCheck('proppr',weightVec,ruleStrings,modeString,inputSymbol,expectedResultDict)

    def _xcompCheck(self,progType,weightVec,ruleStrings,modeString,inputSymbol,expectedResultDict):
        # run the base class check to see that the inference is correct
        if progType=='proppr':
            self.propprInferenceCheck(weightVec,ruleStrings,modeString,inputSymbol,expectedResultDict)
        else:
            self.inferenceCheck(ruleStrings,modeString,inputSymbol,expectedResultDict)
        # setup the next round of tests by compiling a tensorlog
        # Program - this code is lifted from the testtensorlog
        # inference routines
        print 'xcomp inference for mode',modeString,'on input',inputSymbol
        testtensorlog.softmaxNormalize(expectedResultDict)
        rules = parser.RuleCollection()
        for r in ruleStrings:
            rules.add(parser.Parser.parseRule(r))
        if progType=='proppr':
            prog = tensorlog.ProPPRProgram(db=self.db,rules=rules,weights=weightVec)
        else:
            prog = tensorlog.Program(db=self.db,rules=rules)
        mode = declare.ModeDeclaration(modeString)
        tlogFun = prog.compile(mode)
        for compilerClass in TESTED_COMPILERS:
            #cross-compile the function
            xc = compilerClass(prog.db)
            xc.compile(tlogFun)
            # evaluate the theano function and get the output y
            xc.show()
            print '== performing theano eval with',compilerClass,'=='
            ys = xc.evalSymbols([inputSymbol])
            y = ys[0]
            # theano output will a be (probably dense) message, so
            # just compare that maximal elements from these two dicts
            # are the same
            self.checkMaxesInDicts(self.db.rowAsSymbolDict(y), expectedResultDict)
            print '== theano eval checks passed =='
            # cycle through the possible things to differentiate against
            for x in xc.dbMatVar.values() + xc.dbVecVar.values():
                print '== grad with',compilerClass,'wrt',x,'=='
                cost = xc.expr.sum()
                gx, = theano.grad(cost,[x])
                print 'grad cost wrt',x,theano.pp(gx)
                train = theano.function(inputs=xc.exprArgs, outputs=[cost,xc.expr],updates=[(x, (x - 0.1*gx))])
                print '== update function =='
                theano.printing.debugprint(train)
#                print '== compiled function'
#                tmpf = theano.function(inputs=[x],outputs=xc.expr.sum())
#                theano.printing.debugprint(tmpf)
#                #import pudb; pu.db
#                print '== computing gx wrt',x,'type',type(x)
#                gx = theano.grad(xc.expr.sum(),x)
#                print '== gx computed'
#                print theano.pp(gx)
            print '== theano gradients computed =='


    def checkMaxesInDicts(self,actual,expected):
        def maximalElements(d):
            m = max(d.values())
            return set(k for k in d if d[k]==m)
        actualMaxes = maximalElements(actual)
        expectedMaxes = maximalElements(expected)
        print 'actual',actualMaxes,'expected',expectedMaxes
        for a in actualMaxes:
            self.assertTrue(a in expectedMaxes)
        for a in expectedMaxes:
            self.assertTrue(a in actualMaxes)

#        print "\n".join(fun.pprint())
#        y1 = prog.evalSymbols(mode,[inputSymbol])
#        self.checkDicts(self.db.rowAsSymbolDict(y1), expectedResultDict)


if __name__ == "__main__":
    unittest.main()
