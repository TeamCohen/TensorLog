# (C) William W. Cohen and Carnegie Mellon University, 2016

# can call a single test with, e.g.,
# python -m unittest testtensorlog.TestSmallProofs.testIf

# NOTE: google style guide for python is 2-space indents, so I'm
# trying to switch to that - W

import unittest
import logging
import logging.config
import collections
import sys
import math
import os
import os.path
import shutil
import tempfile
import scipy

from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import expt
from tensorlog import funs
from tensorlog import learn
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import parser
from tensorlog import plearn
from tensorlog import program

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__),"test-data/")

def softmax_normalize(expected_result_dict):
  """Compute the softmax of the values in a dictionary, i.e.,
  exponentiate and normalize."""
  for k in expected_result_dict:
    expected_result_dict[k] = math.exp(expected_result_dict[k])
  norm = sum(expected_result_dict.values())
  for c in expected_result_dict:
    expected_result_dict[c] /= norm

def rules_from_strings(rule_strings):
  """Convert a list of strings to a RuleCollection"""
  for r in rule_strings:
    print '>',r
    rules = parser.RuleCollection()
    for r in rule_strings:
      rules.add(parser.Parser.parseRule(r))
  return rules

class DataBuffer(object):
  """ Buffer for learner inputs/outputs, used in testing gradients """
  def __init__(self,db):
    self.db = db
    self.x_symbols = []
    self.y_symbols = []
    self.xs = []
    self.ys = []
  def add_data_symbols(self,sx,symbol_list):
    """symbol_list is a list of symbols that are correct answers to input sx
    for the function associated with the given mode."""
    assert len(symbol_list)>0, 'need to have some desired outputs for each input'
    self.x_symbols.append(sx)
    self.xs.append(self.db.onehot(sx))
    self.y_symbols.append(symbol_list)
    distOverYs = self.db.onehot(symbol_list[0])
    for sy in symbol_list[1:]:
      distOverYs = distOverYs + self.db.onehot(sy)
    distOverYs = distOverYs * (1.0/len(symbol_list))
    self.ys.append(distOverYs)
  def get_data(self):
    """Return matrix pair X,Y - inputs and corresponding outputs of the
    function for the given mode."""
    return self.get_x(),self.get_y()
  def get_x(self):
    assert self.xs, 'no data inserted for mode %r in %r' % (mode,self.xs)
    return mutil.stack(self.xs)
  def get_y(self):
    assert self.ys, 'no labels inserted for mode %r' % mode
    return mutil.stack(self.ys)

def matrixAsTrainingData(db,functor,arity):
    """ Convert a matrix containing pairs x,f(x) to training data for a
    learner.  For each row x with non-zero entries, copy that row
    to Y, and and also append a one-hot representation of x to the
    corresponding row of X.
    """
    xrows = []
    yrows = []
    m = db.matEncoding[(functor,arity)].tocoo()
    n = db.dim()
    for i in range(len(m.data)):
      x = m.row[i]
      xrows.append(scipy.sparse.csr_matrix( ([float(1.0)],([0],[x])), shape=(1,n) ))
      rx = m.getrow(x)
      yrows.append(rx * (float(1.0)/rx.sum()))
    return mutil.stack(xrows),mutil.stack(yrows)


#
# tests
#

class TestModeDeclaration(unittest.TestCase):
  """ Test for mode declarations """

  def test_hash(self):
    d = {}
    m1 = declare.ModeDeclaration('foo(i,o)')
    m2 = declare.ModeDeclaration('foo(i, o)')
    self.assertTrue(m1==m2)
    d[m1] = 1.0
    self.assertTrue(m2 in d)

class TestInterp(unittest.TestCase):
  """Test for interpreter. Doesn't verify output, just executes some
  commands to see if they don't raise errors.
  """

  def setUp(self):
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(TEST_DATA_DIR,"textcattoy.cfacts"),
         "--prog", os.path.join(TEST_DATA_DIR,"textcat.ppr"),
         "--proppr"])
    self.ti = program.Interp(optdict['prog'])
    self.ti.prog.setFeatureWeights()

  def test_list(self):
    self.ti.list("predict/2")
    self.ti.list("predict/io")
    self.ti.list("hasWord/2")
    self.ti.list()
    print self.ti.eval("predict/io", "pb")

class TestSmallProofs(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'fam.cfacts'))

  def test_if(self):
    self.inference_check(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william', {'susan':1.0})

  def test_failure(self):
    self.inference_check(['p(X,Y):-spouse(X,Y).'], 'p(i,o)', 'lottie', {matrixdb.NULL_ENTITY_NAME:1.0})

  def test_reverse_if(self):
    self.inference_check(['p(X,Y):-sister(Y,X).'], 'p(i,o)', 'rachel', {'william':1.0})

  def test_or(self):
    self.inference_check(['p(X,Y):-spouse(X,Y).', 'p(X,Y):-sister(X,Y).'], 'p(i,o)', 'william',
              {'susan':1.0, 'rachel':1.0, 'lottie':1.0, 'sarah':1.0})

  def test_chain(self):
    self.inference_check(['p(X,Z):-spouse(X,Y),sister(Y,Z).'], 'p(i,o)', 'susan',
              {'rachel':1.0, 'lottie':1.0, 'sarah':1.0})
    self.inference_check(['p(X,Z):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
              {'charlotte':1.0, 'lucas':1.0, 'poppy':1.0, 'caroline':1.0, 'elizabeth':1.0})

  def test_mid(self):
    self.inference_check(['p(X,Y):-sister(X,Y),child(Y,Z).'], 'p(i,o)', 'william',
              {'sarah': 1.0, 'rachel': 2.0, 'lottie': 2.0})

  def test_nest(self):
    self.inference_check(['s(X,Y):-spouse(X,Y).','t(X,Z):-spouse(X,Y),s(Y,Z).'], 't(i,o)', 'susan', {'susan': 1.0})

  def test_back1(self):
    self.inference_check(['p(X,Y):-spouse(X,Y),sister(X,Z).'], 'p(i,o)', 'william', {'susan': 3.0})

  def test_back2(self):
    self.inference_check(['p(X,Y):-spouse(X,Y),sister(X,Z1),sister(X,Z2).'],'p(i,o)','william',{'susan': 9.0})

  def test_rec1(self):
    program.DEFAULT_MAXDEPTH=4
    self.inference_check(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 5.0})
    program.DEFAULT_MAXDEPTH=10
    self.inference_check(['p(X,Y):-spouse(X,Y).','p(X,Y):-p(Y,X).'], 'p(i,o)','william',{'susan': 11.0})

  def test_const_output(self):
    self.inference_check(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'sarah', {'william': 1.0})
    self.inference_check(['sis(X,W):-assign(W,william),child(X,Y).'], 'sis(i,o)', 'lottie', {'william': 2.0})

#  TODO: extend bpcompiler so this works
#  def testTrivConstOutput(self):
#    self.inference_check(['sis(X,W):-assign(W,william).'], 'sis(i,o)', 'sarah', {'william': 1.0})
#    self.inference_check(['sis(X,W):-assign(W,william).'], 'sis(i,o)', 'lottie', {'william': 1.0})

  def test_const_chain1(self):
    self.inference_check(['p(X,S) :- assign(S,susan),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

  def test_const_chain2(self):
    self.inference_check(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','sarah',{'pos':1.0})
    self.inference_check(['p(X,Pos) :- assign(Pos,pos),child(X,Y),young(Y).'],'p(i,o)','lottie',{'pos':2.0})

  def test_alt_chain(self):
    self.inference_check(['p(X,W) :- spouse(X,W),sister(X,Y),child(Y,Z).'],'p(i,o)','william',{'susan': 5.0})

  def test_proppr1(self):
    w = 7*self.db.onehot('r1')+3*self.db.onehot('r2')
    self.proppr_inference_check(w,['p(X,Y):-sister(X,Y) {r1}.','p(X,Y):-spouse(X,Y) {r2}.'],'p(i,o)',
                  'william', {'sarah': 7.0, 'rachel': 7.0, 'lottie': 7.0, 'susan': 3.0})
  def test_proppr2(self):
    w = 3*self.db.onehot('r2')
    self.proppr_inference_check(w,['p(X,Y):-spouse(Y,X) {r2}.'],'p(i,o)',
                  'susan', {'william': 3.0})

  def test_reuse1(self):
    self.inference_check(['p(X,Y) :- r(X,Z),r(Z,Y).', 'r(X,Y):-spouse(X,Y).'], 'p(i,o)', 'william',
              {'william':1.0})

  # support routines
  #

  def inference_check(self,rule_strings,mode_string,inputSymbol,expected_result_dict):
    print 'testing inference for mode',mode_string,'on input',inputSymbol,'with rules:'
    softmax_normalize(expected_result_dict)
    rules = rules_from_strings(rule_strings)
    prog = program.Program(db=self.db,rules=rules)
    mode = declare.ModeDeclaration(mode_string)
    fun = prog.compile(mode)
    print "\n".join(fun.pprint())
    y1 = prog.evalSymbols(mode,[inputSymbol])
    self.check_dicts(self.db.rowAsSymbolDict(y1), expected_result_dict)


  def proppr_inference_check(self,weightVec,rule_strings,mode_string,inputSymbol,expected_result_dict):
    print 'testing inference for mode',mode_string,'on input',inputSymbol,'with proppr rules:'
    softmax_normalize(expected_result_dict)
    rules = rules_from_strings(rule_strings)
    prog = program.ProPPRProgram(db=self.db,rules=rules,weights=weightVec)
    mode = declare.ModeDeclaration(mode_string)
    fun = prog.compile(mode)

    y1 = prog.evalSymbols(mode,[inputSymbol])
    self.check_dicts(self.db.rowAsSymbolDict(y1), expected_result_dict)

  def only(self,group):
    self.assertEqual(len(group), 1)
    return group[0]

  def check_dicts(self,actual, expected):
    print 'actual:  ',actual
    if not matrixdb.NULL_ENTITY_NAME in expected:
      expected[matrixdb.NULL_ENTITY_NAME]=0.0
    if expected:
      print 'expected:',expected
      self.assertEqual(len(actual.keys()), len(expected.keys()))
      for k in actual.keys():
        self.assertAlmostEqual(actual[k], expected[k], delta=0.05)


class TestMultiRowOps(unittest.TestCase):
  #TODO document this

  def setUp(self):
    #logging.basicConfig(level=logging.DEBUG)
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'fam.cfacts'))
  def testThing(self):
    self.compareCheck([
    "p(X,Y):-q(X,Y).",
    "p(X,Y):-r(X,Y).",
    "p(X,Y):-spouse(X,Y).",
    "q(X,Y):-child(X,Y).", # DefinedPredOp should return multi-row results
    "r(X,Y):-s(X,Y).",
    "s(X,Y):-r(X,Y).", # NullFunction should return multi-row results
    ],"p(i,o)",["william","rachel"],[
      {#'rachel':1.0,'sarah':1.0,'lottie':1.0,
      'susan':1.0,
      'josh':1.0,'charlie':1.0},
      {'caroline':1.0,'elizabeth':1.0}])
  def compareCheck(self,rule_strings,mode_string,input_symbols,expected_result_dicts):
    for d in expected_result_dicts:
      softmax_normalize(d)
      d[matrixdb.NULL_ENTITY_NAME] = 0
    self.inference_check(rule_strings,mode_string,input_symbols,expected_result_dicts)
    self.predictCheck(rule_strings,mode_string,input_symbols,expected_result_dicts)

  def inference_check(self,rule_strings,mode_string,input_symbols,expected_result_dicts):
    print '\n\ntesting inference for mode',mode_string,'on input',input_symbols,'with rules:'
    for r in rule_strings:
      print '>',r
    rules = parser.RuleCollection()
    for r in rule_strings:
      rules.add(parser.Parser.parseRule(r))
    prog = program.Program(db=self.db,rules=rules)
    mode = declare.ModeDeclaration(mode_string)

    fun = prog.compile(mode)
    for i in range(len(input_symbols)):
      y1 = prog.evalSymbols(mode,[input_symbols[i]])
      self.check_dicts(self.db.rowAsSymbolDict(y1), expected_result_dicts[i])

  def predictCheck(self,rule_strings,mode_string,input_symbols,expected_result_dicts):
    print '\n\ntesting predictions for mode',mode_string,'on input',input_symbols,'with rules:'
    for r in rule_strings:
      print '>',r
    rules = parser.RuleCollection()
    for r in rule_strings:
      rules.add(parser.Parser.parseRule(r))
    prog = program.Program(db=self.db,rules=rules)
    mode = declare.ModeDeclaration(mode_string)

    td = []
    print 'training data:'
    for i in range(len(input_symbols)):
      sol = expected_result_dicts[i].keys()[0]
      td.append("\t".join([mode.functor,input_symbols[i],sol]))
      print td[-1]
    trainingData = self.db.createPartner()
    trainingData.addLines(td)
    trainSpec = (mode.functor,mode.arity)
    X,Y = matrixAsTrainingData(trainingData,*trainSpec)
    learner = learn.OnePredFixedRateGDLearner(prog,epochs=5)
    P0 = learner.predict(mode,X)

  def check_dicts(self,actual, expected):
    print 'actual:  ',actual
    if expected:
      print 'expected:',expected
      self.assertEqual(len(actual.keys()), len(expected.keys()))
      for k in actual.keys():
        self.assertAlmostEqual(actual[k], expected[k], delta=0.0001)

class TestMatrixRecursion(unittest.TestCase):
  """Test that recursion works properly when inputs are mini-batches,
  ie, matrixes instead of one-hot vectors.
  """

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'fam.cfacts'))

  def testRec0(self):
    self.mat_inference_check(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-child(X,Y).'],
                 'r1(i,o)',['william','rachel'],
                 {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})

  def testRec1(self):
    self.mat_inference_check(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-child(X,Y).'],
                 'p(i,o)',['william','rachel'],
                 {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})

  def testRecBound(self):
    self.mat_inference_check(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-spouse(X,Y).'],
                 'p(i,o)',['william','rachel'],
                 {0: {'charlie': 1.0/3.0, 'josh': 1.0/3.0, 'susan':1.0/3.0}, 1: {'caroline': 0.5, 'elizabeth': 0.5}})
    self.mat_inference_check(['p(X,Y):-child(X,Y).','p(X,Y):-r1(X,Y).','r1(X,Y):-spouse(X,Y).'],
                 'p(i,o)',['william','rachel'],
                 {0: {'charlie': 0.5, 'josh': 0.5}, 1: {'caroline': 0.5, 'elizabeth': 0.5}},
                 max_depth=0)

  def mat_inference_check(self,rule_strings,mode_string,input_symbols,expected_result_dict,max_depth=None):
    print 'testing inference for mode',mode_string,'on inputs',input_symbols,'with rules:'
    rules = rules_from_strings(rule_strings)
    prog = program.Program(db=self.db,rules=rules)
    if max_depth!=None: prog.maxDepth=max_depth
    mode = declare.ModeDeclaration(mode_string)
    X = mutil.stack(map(lambda s:prog.db.onehot(s), input_symbols))
    actual = prog.eval(mode,[X])
    print 'compiled functions',prog.function.keys()
    self.check_dict_of_dicts(prog.db.matrixAsSymbolDict(actual), expected_result_dict)

  def check_dict_of_dicts(self,actual,expected):
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

class TestGrad(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'fam.cfacts'))

  def test_if(self):
    rules = ['p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','sarah'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.grad_check(rules, mode, params,
             [('william',['lottie'])],
             {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_if2(self):
    rules = ['p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','sarah']), ('william',['rachel','sarah'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.grad_check(rules, mode, params,
             [('william',['lottie']), ('william',['lottie'])],
             {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_reverse_if(self):
    rules = ['p(X,Y):-parent(Y,X).']
    mode = 'p(i,o)'
    params = [('parent',2)]
    self.grad_check(rules, mode, params,
             [('lottie',['charlotte'])],
             {'parent(charlotte,lottie)': +1,'parent(lucas,lottie)': -1})

  def test_chain1(self):
    rules = ['p(X,Z):-sister(X,Y),child(Y,Z).']
    mode = 'p(i,o)'
    self.grad_check(rules,mode,
             [('sister',2)],
             [('william',['caroline','elizabeth'])],
             {'sister(william,rachel)': +1,'sister(william,lottie)': -1})
    self.grad_check(rules,mode,
             [('child',2)],
             [('william',['caroline','elizabeth'])],
             {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1})

    self.grad_check(rules,mode,
             [('child',2),('sister',2)],
             [('william',['caroline','elizabeth'])],
             {'child(rachel,elizabeth)': +1,'child(lottie,lucas)': -1, 'sister(william,rachel)': +1,'sister(william,lottie)': -1})

  def test_chain2(self):
    rules = ['p(X,Z):-spouse(X,Y),sister(Y,Z).']
    mode = 'p(i,o)'
    self.grad_check(rules,mode,
             [('sister',2)],
             [('susan',['rachel'])],
             {'sister(william,rachel)': +1,'sister(william,lottie)': -1})


  def test_printf(self):
    rules = ['p(X,Z1):-printf(X,X1),spouse(X1,Y),printf(Y,Y1),sister(Y1,Z),printf(Z,Z1).']
    mode = 'p(i,o)'
    self.grad_check(rules,mode,
             [('sister',2)],
             [('susan',['rachel'])],
             {'sister(william,rachel)': +1,'sister(william,lottie)': -1})

  def test_call1(self):
    rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-q(Z,W).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','sarah'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.grad_check(rules, mode, params,
             [('william',['lottie'])],
             {'sister(william,rachel)': -1,'sister(william,lottie)': +1})

  def test_call2(self):
    rules = ['q(X,Y):-sister(X,Y).','p(Z,W):-r(Z,W).','r(Z,W):-q(Z,W).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','sarah'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': +1,'sister(william,lottie)': -1})
    self.grad_check(rules, mode, params,
             [('william',['lottie'])],
             {'sister(william,rachel)': -1,'sister(william,lottie)': +1})


  def test_split(self):
    rules = ['p(X,Y):-sister(X,Y),child(Y,Z),young(Z).']
    mode = 'p(i,o)'
    params = [('child',2)]
    self.grad_check(rules, mode, params,
             [('william',['lottie'])],
             {'child(lottie,lucas)': +1,'child(lottie,charlotte)': +1,'child(sarah,poppy)': -1})
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['lottie'])],
             {'sister(william,lottie)': +1,'sister(william,sarah)': -1})

  def test_or(self):
    rules = ['p(X,Y):-child(X,Y).', 'p(X,Y):-sister(X,Y).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['charlie','rachel'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': -1,'sister(william,lottie)': -1})
    params = [('child',2)]
    self.grad_check(rules, mode, params,
             [('william',['charlie','rachel'])],
             {'child(william,charlie)': +1,'child(william,josh)': -1})
    params = [('child',2),('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['charlie','rachel'])],
             {'child(william,charlie)': +1,'child(william,josh)': -1,'sister(william,rachel)': +1,'sister(william,sarah)': -1})


  def test_weighted_vec(self):
    rules = ['p(X,Y):-sister(X,Y),assign(R,r1),feat(R).','p(X,Y):-child(X,Y),assign(R,r2),feat(R).']
    mode = 'p(i,o)'
    params = [('sister',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','charlie'])],
             {'sister(william,rachel)': +1,'sister(william,sarah)': -1})
    params = [('child',2)]
    self.grad_check(rules, mode, params,
             [('william',['rachel','charlie'])],
             {'child(william,charlie)': +1,'child(william,josh)': -1})
    params = [('feat',1)]
    self.grad_check(rules, mode, params,
             [('william',['josh','charlie'])],
             {'feat(r1)': -1,'feat(r2)': +1})
    self.grad_check(rules, mode, params,
             [('william',['rachel','sarah','lottie'])],
             {'feat(r1)': +1,'feat(r2)': -1})

  def grad_check(self,rule_strings,mode_string,params,xyPairs,expected):
    """
    expected - dict mapping strings encoding facts to expected sign of the gradient
    """
    mode = declare.ModeDeclaration(mode_string)
    (prog,updates) = self.grad_updates(rule_strings,mode,params,xyPairs)
    #put the gradient into a single fact-string-indexed dictionary
    updates_with_string_keys = {}
    for (functor,arity),up in updates.items():
      print 'testtensorlog update for',functor,arity,'is',up
      upDict = prog.db.matrixAsPredicateFacts(functor,arity,up)
      print 'upDict',upDict
      for fact,grad_of_fact in upDict.items():
        updates_with_string_keys[str(fact)] = grad_of_fact
    self.check_directions(updates_with_string_keys,expected)

  def check_directions(self,actual_grad,expected_direction):
    #TODO allow expected to contain zeros?
    for fact,sign in expected_direction.items():
      print fact,'expected sign',sign,'grad',actual_grad.get(fact)
      if not fact in actual_grad: print 'actual_grad',actual_grad
      self.assertTrue(fact in actual_grad)
      self.assertTrue(actual_grad[fact] * sign > 0)

  def grad_updates(self,rule_strings,mode,params,xyPairs):
    """rule_strings - a list of tensorlog rules to use with the db.
    mode_string - mode for the data.
    params - list of (functor,arity) pairs that gradients will be computed for
    xyPairs - list of pairs (x,[y1,..,yk]) such that the desired result for x is uniform dist over y's

    return (program,updates) - updates is a learn.GradAccumulator
    object, which is something like a dictionary mapping parameter
    names to gradients
    """
    #build program
    rules = rules_from_strings(rule_strings)
    prog = program.Program(db=self.db,rules=rules)
    #build dataset
    data = DataBuffer(self.db)
    for x,ys in xyPairs:
      data.add_data_symbols(x,ys)
    #mark params: should be pairs (functor,arity)
    prog.db.clearParameterMarkings()
    for functor,arity in params:
      prog.db.markAsParameter(functor,arity)
    #compute gradient
    learner = learn.OnePredFixedRateGDLearner(prog)
    updates = learner.crossEntropyGrad(mode,data.get_x(),data.get_y())
    return prog,updates

class TestProPPR(unittest.TestCase):

  def setUp(self):
    #logging.basicConfig(level=logging.DEBUG)
    self.prog = program.ProPPRProgram.load(
        [os.path.join(TEST_DATA_DIR,'textcat.ppr'),
         os.path.join(TEST_DATA_DIR,'textcattoy.cfacts')])
    self.labeledData = self.prog.db.createPartner()
    def moveToPartner(db,partner,functor,arity):
      partner.matEncoding[(functor,arity)] = db.matEncoding[(functor,arity)]
      if (functor,arity) in self.prog.getParamList():
        partner.params.add((functor,arity))
        db.paramSet.remove((functor,arity))
        db.paramList.remove((functor,arity))
      del db.matEncoding[(functor,arity)]
    moveToPartner(self.prog.db,self.labeledData,'train',2)
    moveToPartner(self.prog.db,self.labeledData,'test',2)
    self.prog.setFeatureWeights()
    self.xsyms,self.X,self.Y = self.loadExamples(
        os.path.join(TEST_DATA_DIR,'textcattoy-train.examples'),
        self.prog.db)
    self.numExamples = self.X.get_shape()[0]
    self.numFeatures = self.X.get_shape()[1]
    self.mode = declare.ModeDeclaration('predict(i,o)')
    self.numWords = \
      {'dh':4.0, 'ft':5.0, 'rw':3.0, 'sc':5.0, 'bk':5.0,
       'rb':4.0, 'mv':8.0,  'hs':9.0, 'ji':6.0, 'tf':8.0, 'jm':8.0 }
    self.rawPos = "dh ft rw sc bk rb".split()
    self.rawNeg = "mv hs ji tf jm".split()
    self.rawData = {
        'dh': 'a pricy doll house',
        'ft': 'a little red fire truck',
        'rw': 'a red wagon',
        'sc': 'a pricy red sports car',
        'bk': 'punk queen barbie and ken',
        'rb': 'a little red bike',
        'mv': 'a big 7-seater minivan with an automatic transmission',
        'hs': 'a big house in the suburbs with crushing mortgage',
        'ji': 'a job for life at IBM',
        'tf': 'a huge pile of tax forms due yesterday',
        'jm': 'huge pile of junk mail bills and catalogs'}

  def testDBKeys(self):
    pass
    # symbol table is now hidden and more complicated due to types
    # self.assertTrue(self.prog.db.stab.hasId(matrixdb.NULL_ENTITY_NAME))

  def testNativeRow(self):
    for i in range(self.numExamples):
      pred = self.prog.eval(self.mode,[self.X.getrow(i)])
      d = self.prog.db.rowAsSymbolDict(pred)
      uniform = {'pos':0.5,'neg':0.5}
      self.check_dicts(d,uniform)

  def testNativeMatrix(self):
    pred = self.prog.eval(self.mode,[self.X])
    d0 = self.prog.db.matrixAsSymbolDict(pred)
    for i,d in d0.items():
      uniform = {'pos':0.5,'neg':0.5,}
      self.check_dicts(d,uniform)

  def testGradMatrix(self):
    data = DataBuffer(self.prog.db)
    X,Y = matrixAsTrainingData(self.labeledData,'train',2)
    learner = learn.OnePredFixedRateGDLearner(self.prog)
    updates =  learner.crossEntropyGrad(declare.ModeDeclaration('predict(i,o)'),X,Y)
    w = updates[('weighted',1)]
    def checkGrad(i,x,psign,nsign):
      ri = w.getrow(i)
      di = self.prog.db.rowAsSymbolDict(ri)
      for toki in self.rawData[x].split():
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

  def testMultiLearn1(self):
    mode = declare.ModeDeclaration('predict(i,o)')
    dset = dataset.Dataset.loadExamples(
        self.prog.db,
        os.path.join(TEST_DATA_DIR,"toytrain.examples"),
        proppr=True)
    for mode in dset.modesToLearn():
      X = dset.getX(mode)
      Y = dset.getY(mode)
      print mode
      print "\tX "+mutil.pprintSummary(X)
      print "\tY "+mutil.pprintSummary(Y)

    learner = learn.FixedRateGDLearner(self.prog,epochs=5)
    P0 = learner.datasetPredict(dset)
    acc0 = learner.datasetAccuracy(dset,P0)
    xent0 = learner.datasetCrossEntropy(dset,P0)
    print 'toy train: acc0',acc0,'xent1',xent0

    learner.train(dset)

    P1 = learner.datasetPredict(dset)
    acc1 = learner.datasetAccuracy(dset,P1)
    xent1 = learner.datasetCrossEntropy(dset,P1)
    print 'toy train: acc1',acc1,'xent1',xent1

    self.assertTrue(acc0<acc1)
    self.assertTrue(xent0>xent1)
    self.assertTrue(acc1==1)

    Udset = dataset.Dataset.loadExamples(
        self.prog.db,
        os.path.join(TEST_DATA_DIR,"toytest.examples"),
        proppr=True)

    P2 = learner.datasetPredict(Udset)
    acc2 = learner.datasetAccuracy(Udset,P2)
    xent2 = learner.datasetCrossEntropy(Udset,P2)
    print 'toy test: acc2',acc2,'xent2',xent2

    self.assertTrue(acc2==1)
    ##


  def testLearn(self):
    mode = declare.ModeDeclaration('predict(i,o)')
    X,Y = matrixAsTrainingData(self.labeledData,'train',2)
    learner = learn.OnePredFixedRateGDLearner(self.prog,epochs=5)
    P0 = learner.predict(mode,X)
    acc0 = learner.accuracy(Y,P0)
    xent0 = learner.crossEntropy(Y,P0,perExample=True)

    learner.train(mode,X,Y)
    P1 = learner.predict(mode,X)
    acc1 = learner.accuracy(Y,P1)
    xent1 = learner.crossEntropy(Y,P1,perExample=True)

    self.assertTrue(acc0<acc1)
    self.assertTrue(xent0>xent1)
    self.assertTrue(acc1==1)
    print 'toy train: acc1',acc1,'xent1',xent1

    TX,TY = matrixAsTrainingData(self.labeledData,'test',2)
    P2 = learner.predict(mode,TX)
    acc2 = learner.accuracy(TY,P2)
    xent2 = learner.crossEntropy(TY,P2,perExample=True)
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

  def check_dicts(self,actual, expected, delta=0.05):
    if not matrixdb.NULL_ENTITY_NAME in actual:
      actual[matrixdb.NULL_ENTITY_NAME]=0.0
    print 'actual:  ',actual
    if expected:
      if not matrixdb.NULL_ENTITY_NAME in expected:
        expected[matrixdb.NULL_ENTITY_NAME]=0.0
      print 'expected:',expected
      self.assertEqual(len(actual.keys()), len(expected.keys()))
      for k in actual.keys():
        self.assertAlmostEqual(actual[k], expected[k], delta=0.05)



class TestExpt(unittest.TestCase):

  def setUp(self):
    self.cacheDir = tempfile.mkdtemp()

  def cacheFile(self,fileName):
    return os.path.join(self.cacheDir,fileName)

  def testMToyExpt(self):
    acc,xent = self.runMToyExpt1()
    acc,xent = self.runMToyExpt2()
    acc,xent = self.runMToyExpt3()
    #TODO check performance

  def testMToyParallel(self):
    acc,xent = self.runMToyParallel()

  def testTCToyExpt(self):
    #test serialization and uncaching by running the experiment 2x
    acc1,xent1 = self.runTCToyExpt()
    acc2,xent2 = self.runTCToyExpt()
    print 'acc:',acc1,'/',acc2,'xent',xent1,'/',xent2
    self.assertAlmostEqual(acc1,acc2)
    self.assertAlmostEqual(acc1,1.0)
    self.assertAlmostEqual(xent1,xent2)

  def testTCToyExpt2(self):
    #test serialization and uncaching by running the experiment 2x
    acc1,xent1 = self.runTCToyExpt2()
    print 'acc:',acc1,'xent',xent1
    self.assertAlmostEqual(acc1,1.0)

  def runTCToyExpt(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('textcat.db'),
        str(os.path.join(TEST_DATA_DIR,'textcattoy.cfacts')))
    db.listing()
    trainData = dataset.Dataset.uncacheMatrix(self.cacheFile('train.dset'),db,'predict/io','train')
    testData = dataset.Dataset.uncacheMatrix(self.cacheFile('test.dset'),db,'predict/io','test')
    print 'trainData:\n','\n'.join(trainData.pprint())
    print 'testData"\n','\n'.join(testData.pprint())
    prog = program.ProPPRProgram.load(
        [os.path.join(TEST_DATA_DIR,"textcat.ppr")],
        db=db)
    prog.setFeatureWeights()
    params = {'prog':prog,
              'trainData':trainData, 'testData':testData,
              'savedModel':self.cacheFile('toy-trained.db'),
              'savedTestPredictions':self.cacheFile('toy-test.solutions.txt'),
              'savedTrainExamples':self.cacheFile('toy-train.examples'),
              'savedTestExamples':self.cacheFile('toy-test.examples')}
    return expt.Expt(params).run()

  def runTCToyExpt2(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('textcat2.db'),
        str(os.path.join(TEST_DATA_DIR,'textcattoy2.cfacts')))
    trainData = dataset.Dataset.uncacheMatrix(self.cacheFile('train.dset'),db,'predict/io','train')
    testData = dataset.Dataset.uncacheMatrix(self.cacheFile('test.dset'),db,'predict/io','test')
    prog = program.ProPPRProgram.load(
        [os.path.join(TEST_DATA_DIR,"textcat2.ppr")],
        db=db)
    prog.setFeatureWeights()
    prog.db.listing()
    params = {'prog':prog,'trainData':trainData, 'testData':testData }
    return expt.Expt(params).run()

  def testTCToyIgnoringTypes(self):
    # experiment with type declarations
    matrixdb.conf.ignore_types = True
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--proppr", "--prog", os.path.join(TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    optdict['prog'].setFeatureWeights()
    params = {'prog':optdict['prog'],'trainData':optdict['trainData'], 'testData':optdict['testData']}
    return expt.Expt(params).run()

  def testTCToyTypes(self):
    # experiment with type declarations
    matrixdb.conf.ignore_types = False
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--proppr", "--prog", os.path.join(TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])
    optdict['prog'].setFeatureWeights()
    params = {'prog':optdict['prog'],'trainData':optdict['trainData'], 'testData':optdict['testData']}
    ti = program.Interp(optdict['prog'])
    ti.list("predict/io")
    return expt.Expt(params).run()


  def runMToyExpt1(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('matchtoy.db'),
        str(os.path.join(TEST_DATA_DIR,'matchtoy.cfacts')))
    trainData = dataset.Dataset.uncacheExamples(
        self.cacheFile('mtoy-train.dset'),db,
        os.path.join(TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    testData = trainData
    prog = program.ProPPRProgram.load([os.path.join(TEST_DATA_DIR,"matchtoy.ppr")],db=db)
    prog.setRuleWeights(db.ones())
    params = {'prog':prog,'trainData':trainData, 'testData':testData}
    result = expt.Expt(params).run()
#    for mode in testData.modesToLearn():
#      X = testData.getX(mode)
#      Y = testData.getY(mode)
#      Y_ = prog.eval(mode,[X])
#      print 'mode',mode
#      dX = db.matrixAsSymbolDict(X)
#      dY = db.matrixAsSymbolDict(Y)
#      dY_ = db.matrixAsSymbolDict(Y_)
#      for i in sorted(dX.keys()):
#        print i,'X',dX[i],'Y',dY[i],'Y_',sorted(dY_[i].items(),key=lambda (key,val):-val)
    return result

  def runMToyExpt2(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('matchtoy.db'),
        str(os.path.join(TEST_DATA_DIR,'matchtoy.cfacts')))
    trainData = dataset.Dataset.uncacheExamples(
        self.cacheFile('mtoy-train.dset'),db,
        os.path.join(TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    testData = trainData
    prog = program.ProPPRProgram.load([os.path.join(TEST_DATA_DIR,"matchtoy.ppr")],db=db)
    prog.setRuleWeights(db.ones())
    params = {'prog':prog,'trainData':trainData, 'testData':testData, 'learner':learn.FixedRateSGDLearner(prog)}
    return expt.Expt(params).run()

  def runMToyExpt3(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('matchtoy.db'),
        str(os.path.join(TEST_DATA_DIR,'matchtoy.cfacts')))
    trainData = dataset.Dataset.uncacheExamples(
        self.cacheFile('mtoy-train.dset'),db,
        os.path.join(TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    testData = trainData
    prog = program.ProPPRProgram.load([os.path.join(TEST_DATA_DIR,"matchtoy.ppr")],db=db)
    prog.setRuleWeights(db.ones())
    params = {'prog':prog,'trainData':trainData, 'testData':testData, 'learner':plearn.ParallelAdaGradLearner(prog)}
    return expt.Expt(params).run()

  def runMToyParallel(self):
    db = matrixdb.MatrixDB.uncache(
        self.cacheFile('matchtoy.db'),
        str(os.path.join(TEST_DATA_DIR,'matchtoy.cfacts')))
    trainData = dataset.Dataset.uncacheExamples(
        self.cacheFile('mtoy-train.dset'),db,
        os.path.join(TEST_DATA_DIR,'matchtoy-train.exam'),proppr=False)
    testData = trainData
    prog = program.ProPPRProgram.load(
        [os.path.join(TEST_DATA_DIR,"matchtoy.ppr")],
        db=db)
    prog.setRuleWeights(db.ones())
    params = {'prog':prog,'trainData':trainData,
          'testData':testData,
          'learner':plearn.ParallelFixedRateGDLearner(prog)}
    return expt.Expt(params).run()

class TestDataset(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'matchtoy.cfacts'))

  def testProPPRLoadExamples(self):
    self.checkMatchExamples(os.path.join(TEST_DATA_DIR,'matchtoy-train.examples'), proppr=True)

  def testLoadExamples(self):
    self.checkMatchExamples(os.path.join(TEST_DATA_DIR,'matchtoy-train.exam'), proppr=False)

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
        self.check_dicts(actual,expected)

  def check_dicts(self,actual,expected):
    #print 'actual:  ',actual
    #print 'expected:',expected
    self.assertEqual(len(actual.keys()), len(expected.keys()))
    for k in actual.keys():
      self.assertEqual(actual[k], expected[k])

class TestMatrixUtils(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB.loadFile(os.path.join(TEST_DATA_DIR,'fam.cfacts'))
    self.row1 = self.db.onehot('william')+self.db.onehot('poppy')

  def testRepeat(self):
    mat = mutil.repeat(self.row1,3)
    self.assertEqual(mutil.numRows(mat), 3)
    dm = self.db.matrixAsSymbolDict(mat)
    for i in range(3):
      di = dm[i]
      self.assertTrue('william' in di)
      self.assertTrue('poppy' in di)
      self.assertEqual(len(di.keys()), 2)

class TestTypes(unittest.TestCase):

  def setUp(self):
    self.db = matrixdb.MatrixDB()
    self.testLines = [
        '# :- head(triple,entity)\n',
        '# :- tail(triple,entity)\n',
        '# :-creator(triple,source)\n',
        '# :- rel(triple,relation)\n',
        '\t'.join(['head','rxy','x']) + '\n',
        '\t'.join(['tail','rxy','y']) + '\n',
        '\t'.join(['creator','rxy','nyt']) + '\n',
        '\t'.join(['creator','rxy','fox']) + '\n',
        '\t'.join(['rel','rxy','r']) + '\n',
        '\t'.join(['undeclared','a','b']) + '\n',
    ]
    self.db.addLines(self.testLines)

  def testSerialization(self):
    direc = tempfile.mkdtemp()
    self.db.serialize(direc)
    # replace the database with round-trip deserialization
    self.db = matrixdb.MatrixDB.deserialize(direc)
    self.testStabs()
    self.testDeclarations()

  def testStabs(self):
    expectedSymLists = {
        'source':['__NULL__', '__OOV__', 'nyt', 'fox'],
        'relation':['__NULL__', '__OOV__', 'r'],
        'triple':['__NULL__', '__OOV__', 'rxy'],
        'entity':['__NULL__', '__OOV__', 'x', 'y'],
        matrixdb.THING: ['__NULL__', '__OOV__', 'a', 'b']
    }
    self.assertEqual(len(expectedSymLists.keys()), len(self.db._stab.keys()))
    for typeName in expectedSymLists:
      self.assertTrue(typeName in self.db._stab)
      actualSymList = self.db._stab[typeName].getSymbolList()
      self.assertEqual(len(expectedSymLists[typeName]),len(actualSymList))
      for a,b in zip(expectedSymLists[typeName],actualSymList):
        self.assertEqual(a,b)

  def testDeclarations(self):
    for r in ['head','tail']:
      self.assertEqual(self.db.getDomain(r,2), 'triple')
      self.assertEqual(self.db.getRange(r,2), 'entity')
    for r in ['creator','rel']:
      self.assertEqual(self.db.getDomain(r,2), 'triple')
    self.assertEqual(self.db.getRange('creator',2), 'source')
    self.assertEqual(self.db.getRange('rel',2), 'relation')
    for f in [self.db.getRange,self.db.getDomain]:
      self.assertEqual(f('undeclared',2), matrixdb.THING)

  def testTCToyExptTypes(self):
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(TEST_DATA_DIR,"textcat3.ppr"),
         "--trainData", os.path.join(TEST_DATA_DIR,"toytrain.exam"),
         "--testData", os.path.join(TEST_DATA_DIR,"toytest.exam"),
         "--proppr"])

    prog = optdict['prog']
    prog.setAllWeights()
    expt.Expt({'prog':prog,
              'trainData':optdict['trainData'],
              'testData':optdict['testData'],
               'targetMode':declare.asMode("predict/io")}).run()
    rawInput = prog.db.onehot('pb','doc')
    tmp = prog.eval(declare.asMode('predict/io'),[rawInput])
    viewable = prog.db.rowAsSymbolDict(tmp,'label')
    self.assertTrue(viewable['pos']>viewable['neg'])

class TestTypeSemantics(unittest.TestCase):

  def setUp(self):
    # load typed version of textcat task
    optdict,args = comline.parseCommandLine(
        ["--db", os.path.join(TEST_DATA_DIR,"textcattoy3.cfacts"),
         "--prog", os.path.join(TEST_DATA_DIR,"textcat3.ppr"),
         "--proppr"])
    self.prog = optdict['prog']

  def testTypeInference(self):
    fun = self.prog.compile(declare.asMode("predict/io"))
    self.assertEqual(fun.outputType, "label")

class TestTrainableDeclarations(unittest.TestCase):

  def testIt(self):
    db = matrixdb.MatrixDB()
    db.addLines([
        "# :- trainable(w1,1)\n",
        "# :- trainable(w2,2)\n",
        "# :- trainable(a,b)\n",
        "# :- w1(word)\n",
        "# :- w2(word,word)\n",
        "\t".join(["w1","hello"])+"\n",
        "\t".join(["w2","hello","there"])+"\n"
        ])
    print 'params',db.paramList
    self.assertTrue(db.isParameter(declare.asMode("w1(i)")))
    self.assertTrue(db.isParameter(declare.asMode("w2(i,i)")))
    self.assertFalse(db.isParameter(declare.asMode("trainable(i,i)")))
    w1 = db.matrixAsPredicateFacts('w1',1,db.getParameter('w1',1))
    w2 = db.matrixAsPredicateFacts('w2',2,db.getParameter('w2',2))
    self.assertTrue(len(w1.keys())==1)
    self.assertTrue(str(w1.keys()[0])=="w1(hello)")
    self.assertTrue(len(w2.keys())==1)
    self.assertTrue(str(w2.keys()[0])=="w2(hello,there)")

if __name__=="__main__":
  if len(sys.argv)==1:
    unittest.main()
