import theanoxcomp as xc
import tensorlog
import declare
import testtensorlog
import matrixdb

import unittest
import sys

class TestXCSmallProofs(testtensorlog.TestSmallProofs):
    
    def testIf(self):
        self.inferenceCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

    def testFailure(self):
        #self.inferenceCheck(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'lottie', {matrixdb.NULL_ENTITY_NAME:1.0})
        pass

    def testRevIf(self):
        #self.inferenceCheck(['p(X,Y):-sister(Y,X).'], 'p(i,o)', 'rachel', {'william':1.0})
        pass

    def testOr(self):
        #self.inferenceCheck(['p(X,Y):-spouse(X,Y).', 'p(X,Y):-sister(X,Y).'], 'p(i,o)', 'william', 
        #                    {'susan':1.0, 'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
        pass

    def testChain(self):
#        self.inferenceCheck(['p(X,Z):-spouse(X,Y),sister(Y,Z).'], 'p(i,o)', 'susan', 
#                            {'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
#        self.inferenceCheck(['p(X,Z):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william', 
#                            {'charlotte':1.0, 'lucas':1.0, 'poppy':1.0, 'caroline':1.0, 'elizabeth':1.0})
        pass

    def testMid(self):
#        self.inferenceCheck(['p(X,Y):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william', 
#                            {'sarah': 1.0, 'rachel': 2.0, 'lottie': 2.0})
        pass

    def testNest(self):
#        self.inferenceCheck(['s(X,Y):-spouse(X,Y).','t(X,Z):-spouse(X,Y),s(Y,Z).'], 't(i,o)', 'susan', {'susan': 1.0}) 
        pass

    def testBack1(self):
#        self.inferenceCheck(['p(X,Y):-spouse(X,Y),sister(X,Z).'], 'p(i,o)', 'william', {'susan': 3.0})
        pass

    def testBack2(self):
#        self.inferenceCheck(['p(X,Y):-spouse(X,Y),sister(X,Z1),sister(X,Z2).'],'p(i,o)','william',{'susan': 9.0})
        pass

    def testRec1(self):
#        tensorlog.DEFAULT_MAXDEPTH=4
#        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
#        tensorlog.DEFAULT_MAXDEPTH=10
#        self.inferenceCheck(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 11.0})
        pass

    def testConstOutput(self):
#        self.inferenceCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'sarah', {'william': 1.0})
#        self.inferenceCheck(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'lottie', {'william': 2.0})
        pass

    def testConstChain1(self):
#        self.inferenceCheck(['p(X,S) :- assign(S,susan),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})
        pass

    def testConstChain2(self):
#        self.inferenceCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','sarah',{'pos':1.0})
#        self.inferenceCheck(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','lottie',{'pos':2.0})
        pass

    def testAltChain(self):
#        self.inferenceCheck(['p(X,W) :- spouse(X,W),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})
        pass

    def testProppr1(self):
#        w = 7*self.db.onehot('r1')+3*self.db.onehot('r2')        
#        self.propprInferenceCheck(w,['p(X,Y):-sister(X,Y) {r1}.','p(X,Y):-spouse(X,Y) {r2}.'],'p(i,o)',
#                                  'william', {'sarah': 7.0, 'rachel': 7.0, 'lottie': 7.0, 'susan': 3.0})
        pass

    def testProppr2(self):
#        w = 3*self.db.onehot('r2')
#        self.propprInferenceCheck(w,['p(X,Y):-spouse(Y,X) {r2}.'],'p(i,o)',
#                                  'susan', {'william': 3.0})
        pass

    def testReuse1(self):
#        self.inferenceCheck(['p(X,Y) :- r(X,Z),r(Z,Y).', 'r(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', 
#                            {'william':1.0})
        pass


if __name__ == "__main__":
    if len(sys.argv)==1:
        unittest.main()
    


