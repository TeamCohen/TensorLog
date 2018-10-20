# (C) William W. Cohen and Carnegie Mellon University, 2017

import logging
import numpy as np
import os
import unittest
import sys
import collections
import tempfile

from tensorlog import xctargets

if xctargets.tf:
  import tensorflow as tf
  from tensorlog import tensorflowxcomp
else: 
  tensorflowxcomp=None
if xctargets.theano:
  import theano
  from tensorlog import theanoxcomp
else:
  theanoxcomp=None

from tensorlog import bpcompiler
from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import matrixdb
from tensorlog import learn
from tensorlog import mutil
from tensorlog import parser
from tensorlog import program
from tensorlog import simple
from tensorlog import testtensorlog
from tensorlog import funs
from tensorlog import ops
from tensorlog import learnxcomp as learnxc
from tensorlog.expt import Expt

if xctargets.tf:
  tf.logging.set_verbosity(tf.logging.WARN)
  
TESTED_COMPILERS = []
TESTED_LEARNERS = {}
if xctargets.theano:
  for c in [
    theanoxcomp.DenseMatDenseMsgCrossCompiler,
    theanoxcomp.SparseMatDenseMsgCrossCompiler
    ]:
    TESTED_COMPILERS.append(c)
    TESTED_LEARNERS[c]=theanoxcomp.FixedRateGDLearner
if xctargets.tf:
  for c in [
    tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
    tensorflowxcomp.SparseMatDenseMsgCrossCompiler,
    ]:
    TESTED_COMPILERS.append(c)
    TESTED_LEARNERS[c]=tensorflowxcomp.FixedRateGDLearner
    
RUN_OLD_INFERENCE_TESTS = False
SAVE_SUMMARIES = False

def close_cross_compiler(xc):
  xc.close()
  if xctargets.tf and isinstance(xc,tensorflowxcomp.TensorFlowCrossCompiler):
    tf.reset_default_graph()


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
    ret = dict([ (k,v-e) for (k,v) in list(sdict.items()) if v != e])
    z = sum(ret.values())
    for k in ret: ret[k] = ret[k]/z
    return ret

  def xcomp_check(self,ruleStrings,mode_string,input_symbol,expected_result_dict,compare=False):
    self._xcomp_check('vanilla',None,ruleStrings,mode_string,input_symbol,expected_result_dict,compare)

  def proppr_xcomp_check(self,weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict):
    self._xcomp_check('proppr',weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict)

  def _xcomp_check(self,progType,weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict,compare=False):
    # run the base class check to see that the inference is correct
    if RUN_OLD_INFERENCE_TESTS:
      if progType=='proppr':
        self.proppr_inference_check(weightVec,ruleStrings,mode_string,input_symbol,expected_result_dict)
      else:
        self.inference_check(ruleStrings,mode_string,input_symbol,expected_result_dict)
    # setup the next round of tests by compiling a tensorlog
    # Program - this code is lifted from the testtensorlog
    # inference routines
    print('xcomp inference for mode',mode_string,'on input',input_symbol)
    testtensorlog.softmax_normalize(expected_result_dict)
    rules = parser.RuleCollection()
    for r in ruleStrings:
      rules.add(parser.Parser().parseRule(r))
    if progType=='proppr':
      prog = program.ProPPRProgram(db=self.db,rules=rules,weights=weightVec)
    else:
      prog = program.Program(db=self.db,rules=rules)
    for compilerClass in TESTED_COMPILERS:
      #cross-compile the function
      xc = compilerClass(prog)
      # evaluate the function and get the output y
      #xc.show()
      print('== performing eval with',compilerClass,'==')
      inferenceFun = xc.inferenceFunction(mode_string)
      y = inferenceFun(prog.db.onehot(input_symbol))
      # print 'input',xc.getInputName(mode_string),'args,fun
      # =',xc.inference(mode_string) theano output will a be (probably
      # dense) message, so just compare and check that the maximal
      # elements from these two dicts are the same
      actual_result_dict = self.db.rowAsSymbolDict(y)
      self.check_maxes_in_dicts(actual_result_dict, expected_result_dict)
      # check it's normalized
      l1_error = abs(sum(actual_result_dict.values()) - 1.0)
      #print 'l1_error',l1_error,'actual_result_dict',actual_result_dict,'expected_result_dict',expected_result_dict
      self.assertTrue( l1_error < 0.0001)
      # also test proofCountFun
      proofCountFun = xc.proofCountFunction(mode_string)
      pc = proofCountFun(prog.db.onehot(input_symbol))
      # theano output will a be (probably dense) message, so
      # just compare that maximal elements from these two dicts
      # are the same
      pc_result_dict = self.db.rowAsSymbolDict(pc)
      if len(pc_result_dict)>0:
        self.check_maxes_in_dicts(pc_result_dict, expected_result_dict)
      print('== eval checks passed ==')
      close_cross_compiler(xc)

  def check_maxes_in_dicts(self,actual,expected):
    def maximalElements(d):
      m = max(d.values())
      return set(k for k in d if d[k]==m)
    actualMaxes = maximalElements(actual)
    expectedMaxes = maximalElements(expected)
    print('actual',actualMaxes,'expected',expectedMaxes)
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
    print("XLearner loss/grad eval")
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
        print('learner check for compiler',xc.__class__)
        learner = learnxc.XLearner(prog,xc)
        paramsWithUpdates = learner.crossEntropyGrad(mode,data.get_x(),data.get_y())
        updates_with_string_keys = {}
        for (functor,arity),up in paramsWithUpdates:
          print('testxcomp update for',functor,arity,'is',up)
          upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
          print('upDict',upDict)
          for fact,grad_of_fact in list(upDict.items()):
            # need to flip for cross-compilers
            updates_with_string_keys[str(fact)] = -grad_of_fact
        self.check_directions(updates_with_string_keys,expected)
    

  def xgrad_check(self,rule_strings,mode_string,params,xyPairs,expected):
    print("direct loss/grad eval")
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
        print('grad check for compiler',xc.__class__)
        gradFun = xc.dataLossGradFunction(mode_string)
        updates_with_string_keys = {}
        paramsWithUpdates =  gradFun(data.get_x(),data.get_y())
        for (functor,arity),up in paramsWithUpdates:
          upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
          for fact,grad_of_fact in list(upDict.items()):
            # need to flip for cross-compilers
            updates_with_string_keys[str(fact)] = -grad_of_fact
        self.check_directions(updates_with_string_keys,expected)
    self.learnxc_check(rule_strings,mode_string,params,xyPairs,expected)
    close_cross_compiler(xc)

class TestXCProPPR(testtensorlog.TestProPPR):

  def setUp(self):
    super(TestXCProPPR,self).setUp()
    
  def debug(self):
    return self

  def evalxc(self,xc,input):
    inferenceFun = xc.inferenceFunction('predict/io')
    print(inferenceFun)
    rawPred = inferenceFun(input)
    # trim small numbers to zero
    pred = mutil.mapData(lambda d:np.clip((d - 1e-5),0.00,9999.99), rawPred)
    pred.eliminate_zeros()
    return pred

  def testNativeRow(self):
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      for i in range(self.numExamples):
        pred = self.evalxc(xc, self.X.getrow(i))
        d = self.prog.db.rowAsSymbolDict(pred)
        uniform = {'pos':0.5,'neg':0.5}
        self.check_dicts(d,uniform)
      close_cross_compiler(xc)

  def testNativeMatrix(self):

    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      xc.ensureCompiled(self.mode,inputs=None)
      pred = self.prog.eval(self.mode,[self.X])
      d0 = self.prog.db.matrixAsSymbolDict(pred)
      for i,d in list(d0.items()):
        uniform = {'pos':0.5,'neg':0.5,}
        self.check_dicts(d,uniform)
      close_cross_compiler(xc)

  def testGradVector(self):
    data = testtensorlog.DataBuffer(self.prog.db)
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    learner = learn.OnePredFixedRateGDLearner(self.prog)
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      self.prog.db.markAsParameter('weighted',1)
      #xc.compile(self.mode)
      gradFun = xc.dataLossGradFunction('predict/io')
      for i in range(X.shape[0]):
        print("example",i)
        
        updates =  learner.crossEntropyGrad(declare.ModeDeclaration('predict(i,o)'),X[i],Y[i])
        w0 = updates[('weighted',1)].sum(axis=0)
        print(w0)
        
        updates = gradFun(X[i],Y[i])
        paramKey,w = updates[0]
        print(w)
        # w is different from the w in the corresponding testtensorlog test,
        # which is a crossEntropy gradient for each example, but it should have
        # opposite directions
        nrow,ncol = w.shape
        for i in range(nrow):
          for j in range(ncol):
            self.assertTrue((w[i,j]==0) == (w0[i,j]==0))
            self.assertTrue(w[i,j] * w0[i,j] <= 0)

  def testGradMatrix(self):
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
          self.assertTrue(w[i,j] * w0[i,j] <= 0.0,"i=%d,j=%d,w=%g,w0=%g"%(i,j,w[i,j],w0[i,j]))
      close_cross_compiler(xc)

  def testMultiLearn1(self):
    pass

  def testLearn(self):
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
      print('initial train data loss',loss0)
      TX,TY = testtensorlog.matrixAsTrainingData(self.labeledData,'test',2)
      loss1 = lossFun(TX,TY)
      print('initial test data loss',loss1)
      P = learner.predict('predict/io',X)
      #acc0 = xc.accuracy('predict/io',X,Y)
      acc0 = learner.accuracy(Y,P)
      print('initial train accuracy',acc0)
      TP = learner.predict('predict/io',TX)
      #acc1 = xc.accuracy('predict/io',TX,TY)
      acc1 = learner.accuracy(TY,TP)
      print('initial test accuracy',acc1)

      print('params to optimize',xc.prog.getParamList())
      print('vars to optimize',xc.getParamVariables('predict/io'))
      
#       xc.optimizeDataLoss('predict/io', optimizer, X, Y, epochs=20)
      learner.trainMode('predict/io',X,Y)

      loss2 = lossFun(X,Y)
      print('final train data loss',loss2)
      loss3 = lossFun(TX,TY)
      print('final test data loss',loss3)
      P2 = learner.predict('predict/io',X)
      #acc2 = xc.accuracy('predict/io',X,Y)
      acc2 = learner.accuracy(Y,P2)
      print('final train accuracy',acc2)
      TP2 = learner.predict('predict/io',TX)
      #acc3 = xc.accuracy('predict/io',TX,TY)
      acc3 = learner.accuracy(TY,TP2)
      print('final test accuracy',acc3)


      xc.exportAllLearnedParams()
      v = self.prog.db.getParameter('weighted',1)
      d =  self.prog.db.rowAsSymbolDict(v)
      # sanity check a couple of values
      self.assertTrue(d['little_pos'] > d['little_neg'])
      self.assertTrue(d['big_pos'] < d['big_neg'])
      close_cross_compiler(xc)

      self.assertTrue(acc2>=acc0)
      self.assertTrue(acc3>=acc1)

      self.assertTrue(loss2<loss0)
      self.assertTrue(loss2<loss1)
      
      self.assertTrue(acc2>=0.9)
      self.assertTrue(acc2==1.0)
  
  def testDatasetPredict(self):
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
      
      learner = TESTED_LEARNERS[compilerClass](self.prog,xc=xc,rate=0.1,epochs=20)
      P = learner.predict(mode,X)
      print("X",X.shape)
      print("P",P.shape)
      self.assertTrue(X.shape==P.shape)
      P = learner.datasetPredict(dataset.Dataset({mode:X},{mode:Y}))
      print("X",X.shape)
      print("P",P.getX(mode).shape)
      self.assertTrue(X.shape==P.getX(mode).shape)
      
      return xc,learner,X,Y,P

  def testExptScaffold(self):
    mode = declare.ModeDeclaration('predict(i,o)')
    X,Y = testtensorlog.matrixAsTrainingData(self.labeledData,'train',2)
    TX,TY = testtensorlog.matrixAsTrainingData(self.labeledData,'test',2)
    self.prog.setAllWeights()
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(self.prog)
      learner = TESTED_LEARNERS[compilerClass](self.prog,xc=xc,rate=0.1,epochs=20)
      Expt({'prog':self.prog,
                 'trainData':dataset.Dataset({mode:X},{mode:Y}),
                 'testData':dataset.Dataset({mode:TX},{mode:TY}),
                 'targetMode':mode,
                 'learner':learner
                 }).run()

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testExpt(self):
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
      close_cross_compiler(xc)

class TestXCOpGen(unittest.TestCase):

  # TODO tests for other xcompilers?
  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testTCToyTypes(self):
    matrixdb.conf.ignore_types = False
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
        prog=os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"))
    trainData = tlog.load_small_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"))
    mode = list(trainData.keys())[0]
    docs,labels = trainData[mode]
    xc = tlog.get_cross_compiler()
    ops = xc.possibleOps(docs,'doc')
    print('doc ops',ops)
    self.assertTrue(len(ops)==1)
    (words,wordType) = ops[0]
    self.assertTrue(wordType=='word')
    ops = xc.possibleOps(words,'word')
    self.assertTrue(len(ops)==3)
    pairs = None
    for (expr,exprType) in ops:
      if exprType=='labelWordPair':
        pairs = expr
        break
    self.assertTrue(pairs is not None)
    ops = xc.possibleOps(pairs,'labelWordPair')
    self.assertTrue(len(ops)==2)
    for (expr,exprType) in ops:
      self.assertTrue(exprType=='word')
    close_cross_compiler(xc)

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testTCToyIgnoringTypes(self):
    matrixdb.conf.ignore_types = True
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
        prog=os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"))
    trainData = tlog.load_small_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"))
    mode = list(trainData.keys())[0]
    docs,labels = trainData[mode]
    xc = tlog.get_cross_compiler()
    ops = xc.possibleOps(docs)
    binary_predicates = [functor for (functor,arity) in tlog.db.matEncoding if arity==2]
    self.assertTrue(len(ops) == len(binary_predicates)*2)
    for x in ops:
      # ops should just be tensors
      self.assertFalse(isinstance(x,tuple))
    close_cross_compiler(xc)

class TestXCExpt(unittest.TestCase):


  def testTCToyTypes_wscaffold(self):
    matrixdb.conf.ignore_types = False
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    
    optdict['prog'].setAllWeights()
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(optdict['prog'])
      learner = TESTED_LEARNERS[compilerClass](optdict['prog'],xc)
      Expt({
          'prog':optdict['prog'],
          'trainData':optdict['trainData'],
          'testData':optdict['testData'],
          'learner':learner,
          'targetMode':declare.asMode("predict/io")
          }).run()
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

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testTCToyTypes(self):
    matrixdb.conf.ignore_types = False
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

      # check trainability
      for (functor,arity) in xc.db.matEncoding:
        v = xc.parameterFromDBToVariable(functor,arity)
        if v is not None:
          vIsTrainable = (v in tf.trainable_variables())
          vIsParameter = ((functor,arity) in xc.db.paramSet)
          self.assertEqual(vIsTrainable,vIsParameter)

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
      close_cross_compiler(xc)


  def testTCToyIgnoringTypes_wscaffold(self):
    matrixdb.conf.ignore_types = True
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    optdict['prog'].setAllWeights()
    for compilerClass in TESTED_COMPILERS:
      xc = compilerClass(optdict['prog'])
      learner = TESTED_LEARNERS[compilerClass](optdict['prog'],xc)
      Expt({
          'prog':optdict['prog'],
          'trainData':optdict['trainData'],
          'testData':optdict['testData'],
          'learner':learner,
          'targetMode':declare.asMode("predict/io")
          }).run()
      pbDoc = xc.db.onehot('pb')
      self.checkXC(xc,'predict/io',pbDoc,collections.defaultdict(lambda:191))

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testTCToyIgnoringTypes(self):
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
      close_cross_compiler(xc)

  def checkXC(self,xc,mode,rawInput,expectedCols):
    print('matrixdb.conf.ignore_types',matrixdb.conf.ignore_types)
    db = xc.db
    for (functor,arity),mat in list(db.matEncoding.items()):
      print(functor,arity,'shape',mat.shape)
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
    self.prog = program.ProPPRProgram.loadRules(
        os.path.join(testtensorlog.TEST_DATA_DIR,"matchtoy.ppr"),db=self.db)
    self.dset = dataset.Dataset.loadExamples(
        self.db, os.path.join(testtensorlog.TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    self.prog.setAllWeights()

  def testInScaffold(self):
    print(TESTED_COMPILERS)
    self.assertTrue(self.dset.modesToLearn() > 1)
    self.prog.setAllWeights()
    for compilerClass in TESTED_COMPILERS:
      print(compilerClass)
      xc = compilerClass(self.prog)
      # compile everything
      for mode in self.dset.modesToLearn():
        xc.ensureCompiled(mode)
      learner = TESTED_LEARNERS[compilerClass](self.prog,xc)
      testAcc,testXent = Expt({
          'prog':self.prog,
          'trainData':self.dset,
          'testData':self.dset,
          'learner':learner,
          'savedTestPredictions':'TestMultiModeXC.testInScaffold.%s.solutions.txt'%compilerClass.__name__
          }).run()
      print(testAcc)

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def testIt(self):
    self.assertTrue(self.dset.modesToLearn() > 1)
    for compilerClass in [tensorflowxcomp.DenseMatDenseMsgCrossCompiler,
                          tensorflowxcomp.SparseMatDenseMsgCrossCompiler]:
      xc = compilerClass(self.prog)
      # compile everything
      for mode in self.dset.modesToLearn():
        xc.ensureCompiled(mode,inputs=None)
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
        print('mode',mode,'acc',acc)
      session.close()
      close_cross_compiler(xc)

class TestMatParams(unittest.TestCase):

  def setUp(self):
    self.cacheDir = tempfile.mkdtemp()

  def cacheFile(self,fileName):
    return os.path.join(self.cacheDir,fileName)

  def testMToyMatParam(self):
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"matchtoy.cfacts"),
        prog=os.path.join(testtensorlog.TEST_DATA_DIR,"matchtoy.ppr"))
    trainData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"matchtoy-train.exam"))
    tlog.db.markAsParameter('dabbrev',2)
    factDict = tlog.db.matrixAsPredicateFacts('dabbrev',2,tlog.db.matEncoding[('dabbrev',2)])
    print('before learning',len(factDict),'dabbrevs')
    self.assertTrue(len(factDict)==5)
    for f in sorted(factDict.keys()):
      print('>',str(f),factDict[f])

    # expt pipeline
    mode = list(trainData.keys())[0]
    TX,TY = trainData[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=TY.shape, name='tensorlog/trueY')
    loss = tlog.loss(mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(loss)
    train_batch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    for i in range(5):
      print('epoch',i+1)
      session.run(train_step, feed_dict=train_batch_fd)
    tlog.set_all_db_params_to_learned_values(session)
#    params = {'prog':prog,'trainData':trainData, 'testData':testData}
#    result = expt.Expt(params).run()
#    factDict = db.matrixAsPredicateFacts('dabbrev',2,db.matEncoding[('dabbrev',2)])
#    print 'after learning',len(factDict),'dabbrevs'
#    for f in sorted(factDict.keys()):
#      print '>',str(f),factDict[f]
#    self.assertTrue(len(factDict)>5)

@unittest.skipUnless(xctargets.tf,"Tensorflow not available")
class TestSimple(unittest.TestCase):

  def testEmptyRules(self):
    # should not throw an error
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"))

  def testIncrementalDBLoad(self):
    b = simple.Builder()
    predict,label,hasWord,posPair,negPair = b.predicates("predict,label,hasWord,posPair,negPair")
    doc_t,label_t,word_t,labelWordPair_t = b.types("doc_t,label_t,word_t,labelWordPair_t")
    b.schema += predict(doc_t,label_t) & label(label_t)
    b.schema += hasWord(doc_t,word_t) & posPair(word_t,labelWordPair_t) & negPair(word_t,labelWordPair_t)
    for basename in "textcattoy_corpus.cfacts textcattoy_labels.cfacts textcattoy_pairs.cfacts".split(" "):
      b.db += os.path.join(testtensorlog.TEST_DATA_DIR, basename)
    tlog = simple.Compiler(db=b.db)
    for (functor,arity,nnz) in [('hasWord',2,99),('label',1,2),('negPair',2,56)]:
      m = tlog.db.matEncoding[(functor,arity)]
      self.assertTrue(m.nnz == nnz)

  def testBatch(self):
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
        prog=os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"))
    trainData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"))
    testData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"))
    mode = list(trainData.keys())[0]
    TX,TY = trainData[mode]
    UX,UY = testData[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
    correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}
    loss = tlog.loss(mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(loss)
    train_batch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    acc0 = session.run(accuracy, feed_dict=test_batch_fd)
    print('initial accuracy',acc0)
    self.assertTrue(acc0<0.6)
    for i in range(10):
      print('epoch',i+1)
      session.run(train_step, feed_dict=train_batch_fd)
    acc1 = session.run(accuracy, feed_dict=test_batch_fd)
    print('final accuracy',acc1)
    self.assertTrue(acc1>=0.9)
    # test a round-trip serialization
    # saves the db
    cacheDir = tempfile.mkdtemp()
    db_file = os.path.join(cacheDir,'simple.db')
    tlog.set_all_db_params_to_learned_values(session)
    tlog.serialize_db(db_file)
    # load everything into a new graph and don't reset the learned params
    new_graph = tf.Graph()
    with new_graph.as_default():
      tlog2 = simple.Compiler(
          db=db_file,
          prog=os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"),
          autoset_db_params=False)
      # reconstruct the accuracy measure
      inference2 = tlog2.inference(mode)
      trueY2 = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY2')
      correct2 = tf.equal(tf.argmax(trueY2,1), tf.argmax(inference2,1))
      accuracy2 = tf.reduce_mean(tf.cast(correct2, tf.float32))
      # eval accuracy in a new session
      session2 = tf.Session()
      session2.run(tf.global_variables_initializer())
      test_batch_fd2 = {tlog2.input_placeholder_name(mode):UX, trueY2.name:UY}
      acc3 = session2.run(accuracy2, feed_dict=test_batch_fd2)
      print('accuracy after round-trip serialization',acc3)
      self.assertTrue(acc3>=0.9)
    session.close()

  def testMinibatch(self):
    tlog = simple.Compiler(
        db=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"),
        prog=os.path.join(testtensorlog.TEST_DATA_DIR,"textcat3.ppr"))
    self.runTextCatLearner(tlog)

  def runTextCatLearner(self,tlog):
    trainData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"))
    testData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"))
    mode = list(trainData.keys())[0]
    UX,UY = testData[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
    correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}
    loss = tlog.loss(mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    acc0 = session.run(accuracy, feed_dict=test_batch_fd)
    print('initial accuracy',acc0)
    self.assertTrue(acc0<0.6)
    for i in range(10):
      print('epoch',i+1, end=' ')
      for mode,(TX,TY) in tlog.minibatches(trainData,batch_size=2):
        print('.', end=' ')
        train_minibatch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
        session.run(train_step, feed_dict=train_minibatch_fd)
      print('epoch',i+1,'finished')
    acc1 = session.run(accuracy, feed_dict=test_batch_fd)
    print('final accuracy',acc1)
    self.assertTrue(acc1>=0.9)
    session.close()

  def testBuilder1(self):
    b = simple.Builder()
    X,Y,Z = b.variables("X Y Z")
    aunt,parent,sister,wife = b.predicates("aunt parent sister wife")
    uncle = b.predicate("uncle")
    b += aunt(X,Y) <= uncle(X,Z) & wife(Z,Y)
    b += aunt(X,Y) <= parent(X,Z) & sister(Z,Y)
    r1 = b.rule_id("ruleid_t","r1")
    r2 = b.rule_id("ruleid_t","r2")
    b += aunt(X,Y) <= uncle(X,Z) & wife(Z,Y) // r1
    b += aunt(X,Y) <= parent(X,Z) & sister(Z,Y) // r2
    feature,description = b.predicates("feature description")
    weight = b.predicate("weight")
    F = b.variable("F")
    D = b.variable("D")
    b += aunt(X,Y) <= uncle(X,Z) & wife(Z,Y) // (weight(F) | description(X,D) & feature(X,F))
    b.rules.listing()
    rs = b.rules.rulesFor(parser.Goal('aunt',[X,Y]))
    self.assertEqual(str(rs[0]), "aunt(X,Y) :- uncle(X,Z), wife(Z,Y).")
    self.assertEqual(str(rs[1]), "aunt(X,Y) :- parent(X,Z), sister(Z,Y).")
    self.assertEqual(str(rs[2]), "aunt(X,Y) :- uncle(X,Z), wife(Z,Y) {weight(R1) : assign(R1,r1,ruleid_t)}.")
    self.assertEqual(str(rs[3]), "aunt(X,Y) :- parent(X,Z), sister(Z,Y) {weight(R2) : assign(R2,r2,ruleid_t)}.")
    self.assertEqual(str(rs[4]), "aunt(X,Y) :- uncle(X,Z), wife(Z,Y) {weight(F) : description(X,D),feature(X,F)}.")

  def testBuilder2(self):
    b = simple.Builder()
    predict,assign,weighted,hasWord,posPair,negPair = b.predicates("predict assign weighted hasWord posPair negPair")
    X,Pos,Neg,F,W = b.variables("X Pos Neg F W")
    b += predict(X,Pos) <= assign(Pos,'pos','label') // (weighted(F) | hasWord(X,W) & posPair(W,F))
    b += predict(X,Neg) <= assign(Neg,'neg','label') // (weighted(F) | hasWord(X,W) & negPair(W,F))
    dbSpec = os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts")
    self.runTextCatLearner(simple.Compiler(db=dbSpec,prog=b.rules))

  def testBuilder3(self):
    b = simple.Builder()
    predict,assign,weighted,hasWord,posPair,negPair,label = b.predicates("predict assign weighted hasWord posPair negPair label")
    doc_t,label_t,word_t,labelWordPair_t = b.types("doc_t label_t word_t labelWordPair_t")

    b.schema += predict(doc_t,label_t)
    b.schema += hasWord(doc_t,word_t)
    b.schema += posPair(word_t,labelWordPair_t)
    b.schema += negPair(word_t,labelWordPair_t)
    b.schema += label(label_t)

    X,Pos,Neg,F,W = b.variables("X Pos Neg F W")
    b.rules += predict(X,Pos) <= assign(Pos,'pos','label_t') // (weighted(F) | hasWord(X,W) & posPair(W,F))
    b.rules += predict(X,Neg) <= assign(Neg,'neg','label_t') // (weighted(F) | hasWord(X,W) & negPair(W,F))

    # use the untyped version of the facts to make sure the schema works
    b.db = os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy.cfacts")

    self.runTextCatLearner(simple.Compiler(db=b.db, prog=b.rules))

class TestReparameterizationAndTypedLoading(unittest.TestCase):

  def testBugWasFixed(self):
    # use the untyped version of the facts to make sure the schema works
    db = matrixdb.MatrixDB()
    db.addLines(["# :- r(lo_or_hi_t)\n",
                 "\t".join("r low 0.1".split()) + "\n",
                 "\t".join("r hi 0.9".split()) + "\n"])
    db.markAsParameter('r',1)
    prog = program.Program(db=db)
    typeName = db.schema.getArgType("r",1,0)
    idLow = db.schema.getId(typeName,"low")
    idHi = db.schema.getId(typeName,"hi")
    db_r = db.matEncoding[('r',1)]
    self.approxEqual(db_r[0,idLow], 0.1)
    self.approxEqual(db_r[0,idHi], 0.9)

    xc = tensorflowxcomp.SparseMatDenseMsgCrossCompiler(prog)
    v_r = xc._vector(declare.asMode("r(i)"))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    xc.exportAllLearnedParams()
    print('exported to xc',db.matEncoding[('r',1)])
    db_r = db.matEncoding[('r',1)]
    self.approxEqual(db_r[0,idLow], 0.1)
    self.approxEqual(db_r[0,idHi], 0.9)

  def approxEqual(self,a,b):
    self.assertTrue(abs(float(a)-b) < 0.0001)

class TestPlugins(unittest.TestCase):

  def test_identity_io(self):
    ruleStrings = ['predict(X,Y) :- assign(Pos,pos,label),udp1(Pos,Y) {weighted(F): hasWord(X,W),posPair(W,F)}.',
                   'predict(X,Y) :- assign(Neg,neg,label),udp1(Neg,Y) {weighted(F): hasWord(X,W),negPair(W,F)}.']
    plugins = program.Plugins()
    plugins.define('udp1/io', lambda x:x, lambda inputType:'label')
    self.check_learning_with_udp(ruleStrings,plugins)

  def test_identity_oi(self):
    ruleStrings = ['predict(X,Y) :- assign(Pos,pos,label),udp2(Y,Pos) {weighted(F): hasWord(X,W),posPair(W,F)}.',
                   'predict(X,Y) :- assign(Neg,neg,label),udp2(Y,Neg) {weighted(F): hasWord(X,W),negPair(W,F)}.']
    plugins = program.Plugins()
    plugins.define('udp2/oi', lambda x:x, lambda inputType:'label')
    self.check_learning_with_udp(ruleStrings,plugins)

  def test_double_io1(self):
    ruleStrings = ['predict(X,Y) :- assign(Pos,pos,label),udp3(Pos,Y) {weighted(F): hasWord(X,W),posPair(W,F)}.',
                   'predict(X,Y) :- assign(Neg,neg,label),udp3(Neg,Y) {weighted(F): hasWord(X,W),negPair(W,F)}.']
    plugins = program.Plugins()
    plugins.define('udp3/io', lambda x:2*x, lambda inputType:'label')
    self.check_learning_with_udp(ruleStrings,plugins)

  def test_double_io2(self):
    ruleStrings = ['predict(X,Pos) :- assign(Pos,pos,label) {weighted(F): hasWord(X,W),double(W,W2),posPair(W2,F)}.',
                   'predict(X,Neg) :- assign(Neg,neg,label) {weighted(F2): hasWord(X,W),negPair(W,F),double(F,F2)}.']
    plugins = program.Plugins()
    plugins.define('double/io', lambda x:2*x, lambda inputType:inputType)
    self.check_learning_with_udp(ruleStrings,plugins)

  def test_kw_i(self):
    ruleStrings = ['predict(X,Pos) :- assign(Pos,pos,label),hasWord(X,W),poskw(W).',
                   'predict(X,Neg) :- assign(Neg,neg,label),hasWord(X,W),negkw(W).']
    plugins = program.Plugins()
    db = matrixdb.MatrixDB.loadFile(os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"))
    poskw_v = (db.onehot('little','word') + db.onehot('red','word')).todense()
    negkw_v = (db.onehot('big','word') + db.onehot('job','word') + db.onehot('huge','word')).todense()
    plugins.define('poskw/i', lambda:poskw_v, lambda:'word')
    plugins.define('negkw/i', lambda:negkw_v, lambda:'word')
    self.check_udp(ruleStrings,plugins)
   
  def check_udp(self,ruleStrings,plugins):
    db = matrixdb.MatrixDB.loadFile(os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts"))
    rules = testtensorlog.rules_from_strings(ruleStrings)
    prog = program.ProPPRProgram(rules=rules,db=db,plugins=plugins)
    mode = declare.asMode("predict/io")
    prog.compile(mode)
    fun = prog.function[(mode,0)]
    print("\n".join(fun.pprint()))
    tlog = simple.Compiler(db=db, prog=prog)
    testData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"))
    mode = list(testData.keys())[0]
    UX,UY = testData[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
    correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    acc1 = session.run(accuracy, feed_dict=test_batch_fd)
    print('final accuracy',acc1)
    session.close()


  # TOFIX needs some work to pass
  # - you can't do polytree BP with multiple inputs
  # - so there's not a simple fix
  # - probably do this: (1) treat inputs to leftmost userDef as outputs (2) run message-passing for those outputs
  # (3) add the user def operator (4) repeat .... (5) when there are no more plugins
  def notest_isect_iio(self):
    bpcompiler.conf.trace = True
    ruleStrings = ['predict(X,Y) :- hasWord(X,W),posPair(W,P1),negPair(W,P2),isect(P1,P2,Y).']
    plugins = program.Plugins()
    plugins.define('isect/iio', lambda x1,x2:x1*x2, lambda t1,t2:t1)
    self.assertTrue(plugins.isDefined(declare.asMode('isect/iio')))
    self.check_learning_with_udp(ruleStrings,plugins)
    
  def argmax(self):
    bpcompiler.conf.trace = True
    ruleStrings = ['predict(X,Y):-olympics(X,Z),nations(Z),argmax(Z,Y).']
    plugins = program.Plugins()
    plugins.define('argmax/io',lambda x1:tf.nn.softmax(x1), lambda t1:t1)
    db = matrixdb.MatrixDB.loadFile(os.path.join(testtensorlog.TEST_DATA_DIR,'argmax.cfacts'))
    rules = testtensorlog.rules_from_strings(ruleStrings)
    prog = program.ProPPRProgram(rules=rules,db=db,plugins=plugins)
    prog.setAllWeights()
    mode = declare.asMode("predict/io")
    prog.compile(mode)
    fun = prog.function[(mode,0)]
    print("\n".join(fun.pprint()))
    tlog = simple.Compiler(db=db, prog=prog)
    
    data = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"argmax.exam"))
    mode = list(data.keys())[0]
    UX,UY = data[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
    correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    acc0 = session.run(accuracy, feed_dict=test_batch_fd)
    print('initial accuracy',acc0)
    self.assertTrue(acc0>0.9)
    session.close()
    
#     acc0 = session.run(inference, feed_dict=test_batch_fd)
#     print "inference results:"
#     print acc0
#     print np.argmax(acc0,1)
#     print "trueY:"
#     print UY
#     print np.argmax(UY,1)

  @unittest.skipUnless(xctargets.tf,"Tensorflow not available")
  def check_learning_with_udp(self,ruleStrings,plugins,dbfile=os.path.join(testtensorlog.TEST_DATA_DIR,"textcattoy3.cfacts")):
    db = matrixdb.MatrixDB.loadFile(dbfile)
    rules = testtensorlog.rules_from_strings(ruleStrings)
    prog = program.ProPPRProgram(rules=rules,db=db,plugins=plugins)
    prog.setAllWeights()
    mode = declare.asMode("predict/io")
    prog.compile(mode)
    fun = prog.function[(mode,0)]
    print("\n".join(fun.pprint()))
    tlog = simple.Compiler(db=db, prog=prog)

    trainData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytrain.exam"))
    testData = tlog.load_dataset(os.path.join(testtensorlog.TEST_DATA_DIR,"toytest.exam"))
    mode = list(trainData.keys())[0]
    TX,TY = trainData[mode]
    UX,UY = testData[mode]
    inference = tlog.inference(mode)
    trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
    correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}
    loss = tlog.loss(mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_step = optimizer.minimize(loss)
    train_batch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    acc0 = session.run(accuracy, feed_dict=test_batch_fd)
    print('initial accuracy',acc0)
    self.assertTrue(acc0<0.6)
    for i in range(10):
      print('epoch',i+1)
      session.run(train_step, feed_dict=train_batch_fd)
    acc1 = session.run(accuracy, feed_dict=test_batch_fd)
    print('final accuracy',acc1)
    self.assertTrue(acc1>=0.9)
    session.close()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  # default is to test on everything adding command line arguments
  # 'tensorflow' 'theano' 'sparse' 'dense' filters the list (so
  # 'testxcomp.py tensorflow sparse' will run just
  # tensorflowxcomp.SparseMatDenseMsgCrossCompiler)

  if 'theano' in sys.argv[1:]:
    TESTED_COMPILERS = [c for c in TESTED_COMPILERS if c.__module__.endswith("theanoxcomp")]
  if 'tensorflow' in sys.argv[1:]:
    TESTED_COMPILERS = [c for c in TESTED_COMPILERS if c.__module__.endswith("tensorflowxcomp")]
  if 'dense' in sys.argv[1:]:
    TESTED_COMPILERS = [c for c in TESTED_COMPILERS if c.__name__.startswith("Dense")]
  if 'sparse' in sys.argv[1:]:
    TESTED_COMPILERS = [c for c in TESTED_COMPILERS if c.__name__.startswith("Sparse")]
  sys.argv = [a for a in sys.argv if a not in "theano tensorflow dense sparse".split()]
  print('TESTED_COMPILERS',TESTED_COMPILERS)
  
  unittest.main()
