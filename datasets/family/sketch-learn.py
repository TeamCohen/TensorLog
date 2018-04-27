import os
import sys
import getopt
import scipy.sparse as SS
import scipy.io
import numpy as np
from tensorlog import expt,dataset,comline,matrixdb,mutil,program,funs,sketchcompiler,learn,declare
from tensorlog.helper.fast_sketch import FastSketcher2# as DefaultSketcher
from tensorlog.helper.fast_sketch import FastSketcher# as DefaultSketcher
from tensorlog.helper.countmin_embeddings import Sketcher# as DefaultSketcher
from tensorlog.helper.countmin_embeddings import Sketcher2# as DefaultSketcher
import probe as p
import random
random.seed(3141592)

SKETCHERS={
    "FastSketcher2":FastSketcher2,
    "FastSketcher":FastSketcher,
    "Sketcher":Sketcher,
    "Sketcher2":Sketcher2
    }

def sketch_dataset(dset,sk):
    xDict = {}
    yDict = {}
    for mode in dset.modesToLearn():
        try:
            xDict[mode] = sk.sketch(dset.getX(mode))
            yDict[mode] = sk.sketch(dset.getY(mode))
        except:
            print mode
            raise
    return dataset.Dataset(xDict,yDict)

stem = "kinship"
def setExptParams():
    db = comline.parseDBSpec('tmp-cache/{stem}.db|inputs/{stem}.cfacts:inputs/{stem}-rule.cfacts'.format(stem=stem))
    trainData = comline.parseDatasetSpec('tmp-cache/{stem}-train.dset|inputs/{stem}-train.examples'.format(stem=stem),db)
    testData = comline.parseDatasetSpec('tmp-cache/{stem}-test.dset|inputs/{stem}-test.examples'.format(stem=stem),db)
    #print 'train:','\n  '.join(trainData.pprint())
    #print 'test: ','\n  '.join(testData.pprint())

    optlist,args = getopt.getopt(sys.argv[1:],'x',"k= delta= sketcher=".split())
    optdict = dict(optlist)
    print 'optdict',optdict
    k=optdict.get('--k','160')
    delta = optdict.get('--delta','0.01')
    sketcher_s = optdict.get('--sketcher','FastSketcher2')
    if sketcher_s not in SKETCHERS:
        print "no sketcher named %s" % sketcher_s
        print "available sketchers:"
        print "\n".join(SKETCHERS.keys())
        exit(0)
    sketcher_c = SKETCHERS[sketcher_s]
    
    # 'weighted' has outdegree 151 and we're getting too few
    # values out of unsketch if k is too low (?)
    sketcher = sketcher_c(db,int(k),float(delta)/db.dim(),verbose=False)
    sketcher.describe()

    if probe:
        trainData = trainData.extractMode(mode)
        testData = testData.extractMode(mode)
    
    skTrain = SketchData(trainData,sketcher,"train")
    skTest = SketchData(testData,sketcher,"test")
    
    prog = program.ProPPRProgram.loadRules("theory.ppr",db=db)
    prog.compilerDef = lambda m,p,d,r: sketchcompiler.SketchCompiler(m,p,d,r,sketcher)
    prog.softmaxDef = lambda f:sketchcompiler.SketchSoftmaxFunction(f,sketcher)
    prog.setRuleWeights()
    prog.maxDepth=4
    return (prog, skTrain,skTest, sketcher, (k,delta,sketcher_s))

mode = declare.asMode("i_niece/io")
probe = False

def runMain():
    if not os.path.exists("tmp-cache"): os.mkdir("tmp-cache")
    (prog, trainData, testData, sk, args) = setExptParams()

    if not probe:
        print accExpt(prog,trainData,testData,sk,args)
        return (None, prog, testData, mode, sk)
    else:
        insert_sk_funs(prog.getFunction(mode),sk)
        learner = SketchLearner(sk,prog,epochs=4)
        learner.listen.append(trainData)
        learner.listen.append(testData)
        p.probe(learner,prog,testData,mode)
        return (learner,prog,testData,mode,sk)

def insert_sk_funs(fun,sk):
    if hasattr(fun,'sk'):return
    fun.sk = sk
    if hasattr(fun,'fun'): insert_sk_funs(fun.fun,sk)
    if hasattr(fun,'funs'):
        for f in fun.funs: insert_sk_funs(f,sk)
    
def accExpt(prog,trainData,testData,sketcher,args):
    learner = SketchLearner(sketcher,prog,epochs=10)
    learner.listen.append(trainData)
    learner.listen.append(testData)
    params = {'prog':prog,
              'trainData':trainData,
              'testData':testData,
              'savedModel':'tmp-cache/%s-trained.db' % stem,
              'savedTestPredictions':'tmp-cache/%s-test.solutions.%s.txt' % (stem,"-".join(args)),
              'savedTrainExamples':'tmp-cache/%s-train.examples' % stem,
              'savedTestExamples':'tmp-cache/%s-test.examples' % stem,
              'learner':learner,
    }
    return expt.Expt(params).run()

class SketchData(dataset.Dataset):
    def __init__(self,image,sketcher,name):
        self.native = image
        self.name=name
        self.sketch = sketch_dataset(self.native,sketcher)
        self.toggle('XY','sketch')
    def toggle(self,vars,version):
        for var in vars:
            if var == 'Y':
                if version == 'native':
                    self.yDict = self.native.yDict
                elif version == 'sketch':
                    self.yDict = self.sketch.yDict
                else:
                    assert False, "Bad version %s" % version
            elif var == 'X':
                if version == 'native':
                    self.xDict = self.native.xDict
                elif version == 'sketch':
                    self.xDict = self.sketch.xDict
                else:
                    assert False, "Bad version %s" % version
            else:
                assert False, "Bad var %s" % var

class SketchLearner(learn.FixedRateGDLearner):
    def __init__(self,sketcher,prog,**kwargs):
        super(SketchLearner,self).__init__(prog,**kwargs)
        self.sketcher = sketcher
        self.listen=[]
        self.predict = self.native_predict
        #self.state = "pretrain"
    def native_predict(self,mode,X,pad=None):
        P=super(SketchLearner,self).predict(mode,X,pad)
        UP=self.sketcher.unsketch(P)
        #print mode,mutil.pprintSummary(P),mutil.pprintSummary(UP)
        return UP
    def sketch_predict(self,mode,X,pad=None):
        return super(SketchLearner,self).predict(mode,X,pad)
    def train(self,dset):
        self.predict=self.sketch_predict
        for a in self.listen: a.toggle('XY','sketch')
        super(SketchLearner,self).train(dset)
    def datasetPredict(self,dset,copyXs=True):
        self.predict=self.native_predict
        for a in self.listen:
            a.toggle('X','sketch')
            a.toggle('Y','native')
        result = super(SketchLearner,self).datasetPredict(dset,copyXs)
        for a in self.listen:
            a.toggle('X','native')
        result.xDict = dset.xDict
        return result
    def applyUpdate(self,paramGrads,rate):
        """Add each gradient to the appropriate param, after scaling by rate,
        and clip negative parameters to zero.
        """ 
        paramGrads.fitParameterShapes()
        for (functor,arity),delta in paramGrads.items():
            m0 = self.prog.db.getParameter(functor,arity)
            try:
              m1 = m0 + rate * self.sketcher.unsketch(delta)
            except ValueError:
              print mutil.pprintSummary(m0)
              print rate
              print mutil.pprintSummary(delta)
              raise
            m2 = mutil.mapData(lambda d:np.clip(d,0.0,np.finfo('float32').max), m1)
            self.prog.db.setParameter(functor,arity,m2)


if __name__=="__main__":
    data = runMain()
