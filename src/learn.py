# (C) William W. Cohen and Carnegie Mellon University, 2016

import funs
import tensorlog

import scipy.sparse as SS
import numpy.random as NR

class Learner(object):

    # prog pts to db, rules
    def __init__(self,prog):
        self.prog = prog
        self.xSyms = []
        self.ySyms = []
        #TODO functions for these
        self.empiricalLoss = None
#       self.regularizationLoss = None

    def addData(self,mode,filename):
        #TODO allow data for multiple modes, maybe one mode per file
        self.mode = mode
        xs = []
        ys = []
        for line in open(filename):
            sx,sy = line.strip().split("\t")
            self.xSyms.append(sx)
            self.ySyms.append(sy)
            xs.append(self.prog.db.onehot(sx))
            ys.append(self.prog.db.onehot(sy))
        self.X = SS.vstack(xs)
        self.Y = SS.vstack(ys)

    def initializeWeights(self):
        ones = self.prog.db.ones()
        n = self.prog.db.dim()
        initWeights = SS.csc_matrix(ones + 0.01*NR.randn(n))
        self.prog.setWeights(initWeights)

    def showMat(self,msg,m):
        dm = self.prog.db.matrixAsSymbolDict(m)
        for r,d in dm.items():
            print msg,r,d
            

    def update(self):
        self.predictFun = self.prog.getPredictFunction(self.mode)
        print 'predictFun',self.predictFun
        P = self.predictFun.eval(self.prog.db, [self.X])
        self.showMat('P',P)
        dPs = self.predictFun.evalGrad(self.prog.db,[self.X])
        dP = dPs[('weighted',1)]
        #self.showMat('dP',dP)
        crossEntFun = funs.CrossEntropy(self.Y,self.predictFun)
        print 'crossEntFun',crossEntFun
        xent = crossEntFun.eval(self.prog.db, [self.X])
        self.showMat('xent',xent)
        dCrossEnt = crossEntFun.evalGrad(self.prog.db, [self.X])[('weighted',1)]
        print 'dCrossEnt',dCrossEnt
        self.showMat('dCrossEnt',dCrossEnt)
        t = crossEntFun.computationTree(self.prog.db,[self.X])
        t.grad()
        t.show()

#        predGrads = self.predictFun.evalGrad(self.prog.db, [self.X])
        #preds is a matrix P where P[i,y] is the normalized score for
        #class y on example i

        #predGrads[w] is a matrix G so that G[i,f] is the gradient of the
        #f-th feature for the prediction for the ....?
#        for w in predGrads:
#            dm = self.prog.db.matrixAsSymbolDict(SS.csr_matrix(predGrads[w]))
#            for r,d in dm.items():
#                print 'grad',w,r,d
#        self.lossFun = funs.CrossEntropy(self.Y,self.predictFun)
#        loss = self.lossFun.eval(self.prog.db, [self.X])
#        print 'training loss',loss
#        gradDict = self.lossFun.evalGrad(self.prog.db, [self.X])
#        print 'training gradient',gradDict
#        for w in gradDict:
#            print 'gradient',w,':',self.prog.db.matrixAsSymbolDict(gradDict[w])


if __name__ == "__main__":
    prog = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
    learner = Learner(prog)
    learner.addData(tensorlog.ModeDeclaration('predict(i,o)'), 'test/textcattoy-train.examples')
    learner.initializeWeights()
    print 'params',learner.prog.db.params
    learner.update()

