import logging
import numpy as np
import os
import unittest
import sys
import collections
from tensorlog.expt import Expt
try:
  import tensorflow as tf
  from tensorlog import tensorflowxcomp
except:
  tf=None
  tensorflowxcomp=None
try:
  import theano
  from tensorlog import theanoxcomp
except:
  theano=None
  theanoxcomp=None

from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import matrixdb
from tensorlog import learn
from tensorlog import mutil
from tensorlog import parser
from tensorlog import program
from tensorlog import testtensorlog
from tensorlog import funs
from tensorlog import ops
from tensorlog import learnxcomp as learnxc
from tensorlog import expt

if tf:
  tf.logging.set_verbosity(tf.logging.WARN)
  

TESTED_COMPILERS = []
TESTED_LEARNERS = {}
if theano:
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    TESTED_COMPILERS.append(c)
    TESTED_LEARNERS[c]=theanoxcomp.FixedRateGDLearner
if tf:
  for c in [
    tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
    tensorflowxcomp.SparseMatDenseMsgCrossCompiler,
    ]:
    TESTED_COMPILERS.append(c)
    TESTED_LEARNERS[c]=tensorflowxcomp.FixedRateGCLearner
    

SAVE_SUMMARIES = False

class TestXCSmallProofs(testtensorlog.TestSmallProofs):

  def test_if(self):
    self.xcomp_check(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

  def test_failure(self):
    self.xcomp_check(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'lottie', {matrixdb.NULL_ENTITY_NAME:1.0})

  def test_reverse_if(self):
    self.xcomp_check(['p(X,Y):-sister(Y,X).'], 'p(i,o)', 'rachel', {'william':1.0})

  def test_or(self):
    self.xcomp_check(['p(X,Y):-spouse(X,Y).', 'p(X,Y):-sister(X,Y).'], 'p(i,o)', 'william',
            {'susan':1.0, 'rachel':1.0, 'lottie':1.0, 'sarah':1.0})

  def test_chain(self):
    self.xcomp_check(['p(X,Z):-spouse(X,Y),sister(Y,Z).'], 'p(i,o)', 'susan',
            {'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
    self.xcomp_check(['p(X,Z):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
            {'charlotte':1.0, 'lucas':1.0, 'poppy':1.0, 'caroline':1.0, 'elizabeth':1.0})

  def test_mid(self):
    self.xcomp_check(['p(X,Y):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
            {'sarah': 1.0, 'rachel': 2.0, 'lottie': 2.0})

  def test_nest(self):
    self.xcomp_check(['s(X,Y):-spouse(X,Y).','t(X,Z):-spouse(X,Y),s(Y,Z).'], 't(i,o)', 'susan', {'susan': 1.0})

  def test_back1(self):
    # fails for tensorflowxcomp
    self.xcomp_check(['p(X,Y):-spouse(X,Y),sister(X,Z).'], 'p(i,o)', 'william', {'susan': 3.0})

  def test_back2(self):
    self.xcomp_check(['p(X,Y):-spouse(X,Y),sister(X,Z1),sister(X,Z2).'],'p(i,o)','william',{'susan': 9.0})

  def test_rec1(self):
    program.DEFAULT_MAXDEPTH=4
    self.xcomp_check(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
    program.DEFAULT_MAXDEPTH=10
    self.xcomp_check(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 11.0})

  def test_const_output(self):
    self.xcomp_check(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'sarah', {'william': 1.0})
    self.xcomp_check(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'lottie', {'william': 2.0})

  def test_const_chain1(self):
    self.xcomp_check(['p(X,S) :- assign(S,susan),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

  def test_const_chain2(self):
    self.xcomp_check(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','sarah',{'pos':1.0})
    self.xcomp_check(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','lottie',{'pos':2.0})

  def test_alt_chain(self):
    self.xcomp_check(['p(X,W) :- spouse(X,W),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})
    pass

  def test_proppr1(self):
    w = 7*self.db.onehot('r1')+3*self.db.onehot('r2')
    self.proppr_xcomp_check(w,['p(X,Y):-sister(X,Y) {r1}.','p(X,Y):-spouse(X,Y) {r2}.'],'p(i,o)',
                'william', {'sarah': 7.0, 'rachel': 7.0, 'lottie': 7.0, 'susan': 3.0})

  def test_proppr2(self):
    w = 3*self.db.onehot('r2')
    self.proppr_xcomp_check(w,['p(X,Y):-spouse(Y,X) {r2}.'],'p(i,o)',
                'susan', {'william': 3.0})

  def test_reuse1(self):
    self.xcomp_check(['p(X,Y) :- r(X,Z),r(Z,Y).', 'r(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william',
            {'william':1.0})

  def _removeZeros(self, sdict):
    if True: return sdict
    e = sdict[None]
    ret = dict([ (k,v-e) for (k,v) in sdict.items() if v != e])
    z = sum(ret.values())
    for k in ret: ret[k] = ret[k]/z
    return ret

  def xcomp_check(self,ruleStrings,mode_string,input_symbol,expected_result_dict,compare=False):
    self._xcomp_check('vanilla',None,ruleStrings,mode_string,input_symbol,expected_result_dict,compare)

  def proppr_xcomp_check(self,weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict):
    self._xcomp_check('proppr',weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict)

  def _xcomp_check(self,progType,weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict,compare=False):
    # run the base class check to see that the inference is correct
    if progType=='proppr':
      self.proppr_inference_check(weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict)
    else:
      self.inference_check(ruleStrings,mode_string,input_symbol,expected_result_dict)
    # setup the next round of tests by compiling a tensorlog
    # Program - this code is lifted from the testtensorlog
    # inference routines
    print 'xcomp inference for mode',mode_string,'on input',input_symbol
    testtensorlog.softmax_normalize(expected_result_dict)
    rules = parser.RuleCollection()
    for r in ruleStrings:
      rules.add(parser.Parser.parseRule(r))
    if progType=='proppr':
      prog = program.ProPPRProgram(db=self.db,rules=rules,weights=weightVec)
    else:
      prog = program.Program(db=self.db,rules=rules)
    for compilerClass in TESTED_COMPILERS:
      #cross-compile the function
      xc = compilerClass(prog)
      # evaluate the function and get the output y
      #xc.show()
      print '== performing eval with',compilerClass,'=='
      inferenceFun = xc.inferenceFunction(mode_string)
      y = inferenceFun(prog.db.onehot(input_symbol))
      # theano output will a be (probably dense) message, so
      # just compare that maximal elements from these two dicts
      # are the same
      self.check_maxes_in_dicts(self.db.rowAsSymbolDict(y), expected_result_dict)
      print '== eval checks passed =='

  def check_maxes_in_dicts(self,actual,expected):
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


class TestXCGrad(testtensorlog.TestGrad):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(testtensorlog.TEST_DATA_DIR,'fam.cfacts'))

  def test_if(self):
    rules = ['p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','sarah'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie'])],
                     {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_if2(self):
    rules = ['p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','sarah']), ('william',['rachel','sarah'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie']), ('william',['lottie'])],
                     {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_reverse_if(self):
    rules = ['p(X,Y):-parent(Y,X).']
    mode = 'p(i,o)'
    params = [('parent',2)]
    self.xgrad_check(rules, mode, params,
                     [('lottie',['charlotte'])],
                     {'parent(charlotte,lottie)': +1,'parent(lucas,lottie)': -1})

  def test_chain1(self):
    rules = ['p(X,Z):-sister(X,Y),child(Y,Z).']
    mode = 'p(i,o)'
    self.xgrad_check(rules,mode,
                     [('sister',2)],
                     [('william',['caroline','elizabeth'])],
                     {'sister(william,rachel)': +1,'sister(william,lottie)': -1})
    self.xgrad_check(rules,mode,
                     [('child',2)],
                     [('william',['caroline','elizabeth'])],
                     {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1})
    self.xgrad_check(rules,mode,
                     [('child',2),('sister',2)],
                     [('william',['caroline','elizabeth'])],
                     {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1, 'sister(william,rachel)': +1,'sister(william,lottie)': -1})

  def test_chain2(self):
    rules =  ['p(X,Z):-spouse(X,Y),sister(Y,Z).']
    mode = 'p(i,o)'
    self.xgrad_check(rules,mode,
                     [('sister',2)],
                     [('susan',['rachel'])],
                     {'sister(william,rachel)': +1,'sister(william,lottie)': -1})


  def test_call1(self):
    rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-q(Z,W).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','sarah'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie'])],
                     {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_call2(self):
    rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-r(Z,W).','r(Z,W):-q(Z,W).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','sarah'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie'])],
                     {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_split(self):
    rules = ['p(X,Y):-sister(X,Y),child(Y,Z),young(Z).']
    mode = 'p(i,o)'
    params = [('child',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie'])],
                     {'child(lottie,lucas)': +1,'child(lottie,charlotte)': +1,'child(sarah,poppy)': -1})
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['lottie'])],
                     {'sister(william,lottie)': +1,'sister(william,sarah)': -1})

  def test_or(self):
    rules = ['p(X,Y):-child(X,Y).', 'p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['charlie','rachel'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': -1,'sister(william,lottie)': -1})
    params = [('child',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['charlie','rachel'])],
                     {'child(william,charlie)': +1,'child(william,josh)': -1})
    params = [('child',2),('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['charlie','rachel'])],
                     {'child(william,charlie)': +1,'child(william,josh)': -1,'sister(william,rachel)': +1,'sister(william,sarah)': -1})


  def test_weighted_vec(self):
    rules = ['p(X,Y):-sister(X,Y),assign(R,r1),feat(R).','p(X,Y):-child(X,Y),assign(R,r2),feat(R).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','charlie'])],
                     {'sister(william,rachel)': +1,'sister(william,sarah)': -1})
    params = [('child',2)]
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','charlie'])],
                     {'child(william,charlie)': +1,'child(william,josh)': -1})
    params = [('feat',1)]
    self.xgrad_check(rules, mode, params,
                     [('william',['josh','charlie'])],
                     {'feat(r1)': -1,'feat(r2)': +1})
    self.xgrad_check(rules, mode, params,
                     [('william',['rachel','sarah','lottie'])],
                     {'feat(r1)': +1,'feat(r2)': -1})

  def learnxc_check(self,rule_strings,mode_string,params,xyPairs,expected):
    print "XLearner loss/grad eval"
    rules = testtensorlog.rules_from_strings(rule_strings)
    prog = program.Program(db=self.db,rules=rules)
    mode = declare.ModeDeclaration(mode_string)
    prog.db.clearParameterMarkings()
    for (functor,arity) in params:
      prog.db.markAsParameter(functor,arity)
    # TODO: not working yet for mini-batches so check each example
    # individually
    for x,ys in xyPairs:
      data = testtensorlog.DataBuffer(self.db)
      data.add_data_symbols(x,ys)
      for compilerClass in TESTED_COMPILERS:
        xc = compilerClass(prog)
        print 'learner check for compiler',xc.__class__
        learner = learnxc.XLearner(prog,xc)
        paramsWithUpdates = learner.crossEntropyGrad(mode,data.get_x(),data.get_y())
        updates_with_string_keys = {}
        for (functor,arity),up in paramsWithUpdates:
          print 'testxcomp update for',functor,arity,'is',up
          upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
          print 'upDict',upDict
          for fact,grad_of_fact in upDict.items():
            # need to flip for cross-compilers
            updates_with_string_keys[str(fact)] = -grad_of_fact
        self.check_directions(updates_with_string_keys,expected)
    

  def xgrad_check(self,rule_strings,mode_string,params,xyPairs,expected):
    print "direct loss/grad eval"
    rules = testtensorlog.rules_from_strings(rule_strings)
    prog = program.Program(db=self.db,rules=rules)
    prog.db.clearParameterMarkings()
    for (functor,arity) in params:
      prog.db.markAsParameter(functor,arity)
    for x,ys in xyPairs:
      data = testtensorlog.DataBuffer(self.db)
      data.add_data_symbols(x,ys)
      for compilerClass in TESTED_COMPILERS:
        xc = compilerClass(prog)
        print 'grad check for compiler',xc.__class__
        gradFun = xc.dataLossGradFunction(mode_string)
        updates_with_string_keys = {}
        paramsWithUpdates =  gradFun(data.get_x(),data.get_y())
        for (functor,arity),up in paramsWithUpdates:
          upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
          for fact,grad_of_fact in upDict.items():
            # need to flip for cross-compilers
            updates_with_string_keys[str(fact)] = -grad_of_fact
        self.check_directions(updates_with_string_keys,expected)
    self.learnxc_check(rule_strings,mode_string,params,xyPairs,expected)

class TestXCProPPR(testtensorlog.TestProPPR):

  def setUp(self):
    super(TestXCProPPR,self).setUp()
    
  def debug(self):
    return self

  def evalxc(self,xc,input):
    inferenceFun = xc.inferenceFunction('predict/io')
    print inferenceFun
    rawPred = inferenceFun(input)
    # trim small numbers to zero
    pred = mutil.mapData(lambda d:np.clip((d - 1e-5),0.00,9999.99), rawPred)
    pred.eliminate_zeros()
    return pred

  def testNativeRow(self):
    #if not tf: return
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      for i in range(self.numExamples):
        pred = self.evalxc(xc, self.X.getrow(i))
        d = self.prog.db.rowAsSymbolDict(pred)
        uniform = {'pos':0.5,'neg':0.5}
        self.check_dicts(d,uniform)

  def testNativeMatrix(self):

    #if not tf: return
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      xc.ensureCompiled(self.mode)
      pred = self.prog.eval(self.mode,[self.X])
      d0 = self.prog.db.matrixAsSymbolDict(pred)
      for i,d in d0.items():
        uniform = {'pos':0.5,'neg':0.5,}
        self.check_dicts(d,uniform)

  def testGradVector(self):
    #if not tf: return
    data = testtensorlog.DataBuffer(self.prog.db)
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    learner = learn.OnePredFixedRateGDLearner(self.prog)
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      self.prog.db.markAsParameter('weighted',1)
      #xc.compile(self.mode)
      gradFun = xc.dataLossGradFunction('predict/io')
      for i in range(X.shape[0]):
        print "example",i
        
        updates =  learner.crossEntropyGrad(declare.ModeDeclaration('predict(i,o)'),X[i],Y[i])
        w0 = updates[('weighted',1)].sum(axis=0)
        print w0
        
        updates = gradFun(X[i],Y[i])
        paramKey,w = updates[0]
        print w
        # w is different from the w in the corresponding testtensorlog test,
        # which is a crossEntropy gradient for each example, but it should have
        # opposite directions
        nrow,ncol = w.shape
        for i in range(nrow):
          for j in range(ncol):
            self.assertTrue((w[i,j]==0) == (w0[i,j]==0))
            self.assertTrue(w[i,j] * w0[i,j] <= 0)

  def testGradMatrix(self):
    #if not tf: return
    data = testtensorlog.DataBuffer(self.prog.db)
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    learner = learn.OnePredFixedRateGDLearner(self.prog)
    updates =  learner.crossEntropyGrad(declare.ModeDeclaration('predict(i,o)'),X,Y)
    w0 = updates[('weighted',1)].sum(axis=0)
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      self.prog.db.markAsParameter('weighted',1)
      #xc.compile(self.mode)
      gradFun = xc.dataLossGradFunction('predict/io')
      updates = gradFun(X,Y)
      paramKey,w = updates[0]
      # w is different from the w in the corresponding testtensorlog test,
      # which is a crossEntropy gradient for each example, but it should have
      # opposite directions
      nrow,ncol = w.shape
      for i in range(nrow):
        for j in range(ncol):
          self.assertTrue((w[i,j]==0) == (w0[i,j]==0),"i=%d,j=%d,w=%g,w0=%g"%(i,j,w[i,j],w0[i,j]))
          self.assertTrue(w[i,j] * w0[i,j] <= 0,"i=%d,j=%d,w=%g,w0=%g"%(i,j,w[i,j],w0[i,j]))

  def testMultiLearn1(self):
    pass

  def testLearn(self):
    #if not tf: return
    mode = declare.ModeDeclaration('predict(i,o)')
    modestr = 'predict/io'
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    for compilerClass in TESTED_COMPILERS:
      self.prog.setRuleWeights()
      self.prog.setFeatureWeights()
      if SAVE_SUMMARIES:
        xc = compilerClass(self.prog,compilerClass.__name__+".summary")
      else:
        xc = compilerClass(self.prog)
      self.prog.db.markAsParameter('weighted',1)
      
      v = self.prog.db.getParameter('weighted',1)
      d =  self.prog.db.rowAsSymbolDict(v)
      # sanity check a couple of values
      self.assertTrue(d['little_pos'] == d['little_neg'])
      self.assertTrue(d['big_pos'] == d['big_neg'])
      
#       optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      learner = TESTED_LEARNERS[compilerClass](self.prog,xc=xc,rate=0.1,epochs=20)

      lossFun = xc.dataLossFunction('predict/io')
      loss0 = lossFun(X,Y)
      print 'initial train data loss',loss0
      TX,TY = testtensorlog.matrixAsTrainingData(self.labeledData,'test',2)
      loss1 = lossFun(TX,TY)
      print 'initial test data loss',loss1
      P = learner.predict('predict/io',X)
      #acc0 = xc.accuracy('predict/io',X,Y)
      acc0 = learner.accuracy(Y,P)
      print 'initial train accuracy',acc0
      TP = learner.predict('predict/io',TX)
      #acc1 = xc.accuracy('predict/io',TX,TY)
      acc1 = learner.accuracy(TY,TP)
      print 'initial test accuracy',acc1

      print 'params to optimize',xc.prog.getParamList()
      print 'vars to optimize',xc.getParamVariables('predict/io')
      
#       xc.optimizeDataLoss('predict/io', optimizer, X, Y, epochs=20)
      learner.trainMode('predict/io',X,Y)

      loss2 = lossFun(X,Y)
      print 'final train data loss',loss2
      loss3 = lossFun(TX,TY)
      print 'final test data loss',loss3
      P2 = learner.predict('predict/io',X)
      #acc2 = xc.accuracy('predict/io',X,Y)
      acc2 = learner.accuracy(Y,P2)
      print 'final train accuracy',acc2
      TP2 = learner.predict('predict/io',TX)
      #acc3 = xc.accuracy('predict/io',TX,TY)
      acc3 = learner.accuracy(TY,TP2)
      print 'final test accuracy',acc3

      self.assertTrue(acc2>=acc0)
      self.assertTrue(acc3>=acc1)
      self.assertTrue(acc2>=0.9)
      self.assertTrue(acc2==1.0)

      self.assertTrue(loss2<loss0)
      self.assertTrue(loss2<loss1)

      xc.exportAllLearnedParams()
      v = self.prog.db.getParameter('weighted',1)
      d =  self.prog.db.rowAsSymbolDict(v)
      # sanity check a couple of values
      self.assertTrue(d['little_pos'] > d['little_neg'])
      self.assertTrue(d['big_pos'] < d['big_neg'])

  def testExptScaffold(self):
    mode = declare.ModeDeclaration('predict(i,o)')
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    TX,TY = testtensorlog.matrixAsTrainingData(self.labeledData,'test',2)
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      learner = TESTED_LEARNERS[compilerClass](self.prog,xc=xc,rate=0.1,epochs=20)
      expt.Expt({'prog':self.prog,
                 'trainData':dataset.Dataset({mode:X},{mode:Y}),
                 'testData':dataset.Dataset({mode:TX},{mode:TY}),
                 'targetMode':mode,
                 'learner':learner
                 }).run()

  def testExpt(self):
    if not tf: return
    mode = declare.ModeDeclaration('predict(i,o)')
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    TX,TY = testtensorlog.matrixAsTrainingData(self.labeledData,'test',2)
    for compilerClass in [tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
                          tensorflowxcomp.SparseMatDenseMsgCrossCompiler]:
      xc = compilerClass(self.prog)
      xc.runExpt(
          prog=self.prog,
          trainData=dataset.Dataset({mode:X},{mode:Y}),
          testData=dataset.Dataset({mode:TX},{mode:TY}),
          targetMode=mode)

class TestXCExpt(unittest.TestCase):

  def testTCToyTypes(self):
    if not tf: return
    matrixdb.conf.ignore_types = False
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    for compilerClass in [tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
                          tensorflowxcomp.SparseMatDenseMsgCrossCompiler]:
      #TESTED_COMPILERS:
      xc = compilerClass(optdict['prog'])
      learner = TESTED_LEARNERS[compilerClass](optdict['prog'],xc)
      expt.Expt({
          'prog':optdict['prog'],
          'trainData':optdict['trainData'],
          'testData':optdict['testData'],
          'learner':learner,
          'targetMode':declare.asMode("predict/io")
          }).run()
      #xc.runExpt(
      #    prog=optdict['prog'],
      #    trainData=optdict['trainData'],
      #    testData=optdict['testData'],
      #    targetMode=declare.asMode("predict/io"))
      pbDoc = xc.db.onehot('pb','doc')
      self.checkXC(xc,'predict/io',pbDoc,{'negPair':115,'posPair':115,'hasWord':59,'weighted':115,'label':5})
      # some checks on the output of pprint
      lines = xc.pprint('predict/io')
      self.assertTrue(lines[0].find("SoftMaxFunction") >= 0)
      self.assertTrue(lines[1].find("SumFunction") >= 0)
      self.assertEqual(len(lines), 16)
      # some checks on misc xcomp API
      self.assertEqual(xc.inferenceOutputType('predict/io'),'label')
      pbId = xc.asSymbolId('pb',typeName='doc')
      pbSym = xc.asSymbol(pbId,typeName='doc')
      self.assertEqual(pbSym,'pb')
      self.assertEqual(xc.asSymbolId('this does not appear in the data',typeName='doc'), -1)

  def testTCToyIgnoringTypes(self):
    if not tf: return
    matrixdb.conf.ignore_types = True
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    for compilerClass in [tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
                          tensorflowxcomp.SparseMatDenseMsgCrossCompiler]:
      xc = compilerClass(optdict['prog'])
      xc.runExpt(
          prog=optdict['prog'],
          trainData=optdict['trainData'],
          testData=optdict['testData'],
          targetMode=declare.asMode("predict/io"))
      pbDoc = xc.db.onehot('pb')
      self.checkXC(xc,'predict/io',pbDoc,collections.defaultdict(lambda:191))


  def checkXC(self,xc,mode,rawInput,expectedCols):
    print 'matrixdb.conf.ignore_types',matrixdb.conf.ignore_types
    db = xc.db
    for (functor,arity),mat in db.matEncoding.items():
      print functor,arity,'shape',mat.shape
      r,c = mat.shape
      self.assertEqual(c,expectedCols[functor])
    inferenceFun = xc.inferenceFunction(mode)
    y = inferenceFun(rawInput)
    r,c = y.shape
    self.assertEqual(c,expectedCols['label'])

class TestMultiModeXC(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(
        os.path.join(testtensorlog.TEST_DATA_DIR,'matchtoy.cfacts'))
    self.prog = program.ProPPRProgram.load(
        [os.path.join(testtensorlog.TEST_DATA_DIR,"matchtoy.ppr")],db=self.db)
    self.dset = dataset.Dataset.loadExamples(
        self.db, os.path.join(testtensorlog.TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    self.prog.setAllWeights()

  def testIt(self):
    if not tf: return
    self.assertTrue(self.dset.modesToLearn() > 1)
    for compilerClass in [tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
                          tensorflowxcomp.SparseMatDenseMsgCrossCompiler]:
      xc = compilerClass(self.prog)
      # compile everything
      for mode in self.dset.modesToLearn():
        xc.ensureCompiled(mode)
      # check the variables
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      session = tf.Session()
      session.run(tf.global_variables_initializer())
      # set up for training
      trainStep = {}
      for mode in self.dset.modesToLearn():
        (dataLossArgs,dataLossExpr) = xc.dataLoss(mode)
        trainStep[mode] = optimizer.minimize(dataLossExpr, var_list=xc.getParamVariables(mode))
      # train
      for i in range(2): #epochs
        for mode in self.dset.modesToLearn():
          X = self.dset.getX(mode)
          Y = self.dset.getY(mode)
          fd = xc.getFeedDict(mode,X,Y,wrapped=False)
          session.run(trainStep[mode],feed_dict=fd)
      # test
      for mode in self.dset.modesToLearn():
        X = self.dset.getX(mode)
        Y = self.dset.getY(mode)
        Y_ = xc.inferenceFunction(mode)(X)
        acc = xc.accuracy(mode,X,Y)
        print 'mode',mode,'acc',acc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv)==1:
        unittest.main()
    else:
        foo=TestXCProPPR('debug')
        foo.setUp()
        bar=foo.debug()
        xc = TESTED_COMPILERS[0](bar.prog)
        inferenceFun = xc.inferenceFunction('predict/io')
        pred = bar.evalxc(xc, bar.X.getrow(0))
        d = bar.prog.db.rowAsSymbolDict(pred)
