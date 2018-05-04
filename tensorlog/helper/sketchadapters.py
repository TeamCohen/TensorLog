
from tensorlog import dataset,learn,mutil,sketchcompiler
import numpy as np
import logging

"""
Sample usage:

from tensorlog.helper.sketchadapters import sketchProgram,SketchData,SketchLearner

sketcher = [some Sketcher() init]
prog = [some Program() init]
trainData_native = [some Dataset() init]
testData_native = [some Dataset() init]

sketchProgram(sketcher, prog)
trainData = SketchData(sketcher, trainData_native)
testData  = SketchData(sketcher, testData_native)
learner   = SketchLearner(sketcher, publish=[trainData,testData], prog, ...)

params = {'prog':prog,
          'trainData':trainData,
          'testData':testData,
          'learner':learner
}
expt.Expt(params).run()
"""

def sketchProgram(sk, prog):
  """Configure a tensorlog Program object to use Sketch-based compile and softmax machinery"""
  prog.compilerDef = lambda m,p,d,r: sketchcompiler.SketchCompiler(m,p,d,r,sk)
  prog.softmaxDef = lambda f:sketchcompiler.SketchSoftmaxFunction(f,sk)
  prog.nullDef = lambda m:sketchcompiler.NullSketchFunction(m,sk)

def sketchDataset(sk, dset):
  """Convert an entity-space Dataset object to a SketchDataset (compatible with SketchLearner)"""
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
  
class SketchData(dataset.Dataset):
    def __init__(self,sketcher,image,name="unnamed"):
        self.sketch = sketchDataset(sketcher,image)
        self.native = image
        self.name=name
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
    def __init__(self,sketcher,publish,prog,**kwargs):
        super(SketchLearner,self).__init__(prog,**kwargs)
        self.sketcher = sketcher
        self.publish = publish
        self.predict = self.native_predict
    def native_predict(self,mode,X,pad=None):
        """Return a prediction vector in entity space"""
        P=super(SketchLearner,self).predict(mode,X,pad)
        UP=self.sketcher.unsketch(P)
        UP.eliminate_zeros()
        #print mode,mutil.pprintSummary(P),mutil.pprintSummary(UP)
        return UP
    def sketch_predict(self,mode,X,pad=None):
        """Return a prediction vector in sketch space"""
        return super(SketchLearner,self).predict(mode,X,pad)
    def train(self,dset):
        self.predict=self.sketch_predict
        for a in self.publish: a.toggle('XY','sketch')
        super(SketchLearner,self).train(dset)
    def datasetPredict(self,dset,copyXs=True):
        assert dset in self.publish, "predicting on a dataset not subscribed to us is not allowed; it screws up the sketched/unsketched assumptions" 
        self.predict=self.native_predict
        for a in self.publish:
            a.toggle('X','sketch')
            a.toggle('Y','native')
        result = super(SketchLearner,self).datasetPredict(dset,copyXs)
        for a in self.publish:
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
              t2 = rate * delta
              m1 = m0 + t2
            except ValueError:
              print "ValueError at %s (%s,%d)" % (str(type(self)),functor,arity)
              print "m0",mutil.pprintSummary(m0)
              print "rate",rate
              print "delta",mutil.pprintSummary(delta)
              print "t2",mutil.pprintSummary(t2)
              raise
            m2 = mutil.mapData(lambda d:np.clip(d,0.0,np.finfo('float32').max), m1)
            self.prog.db.setParameter(functor,arity,m2)
            

# this is only needed for generating sensical trace output
def insert_sk_funs(fun,sk):
    if hasattr(fun,'sk'):return
    fun.sk = sk
    for f in fun.children(): insert_sk_funs(f,sk)
