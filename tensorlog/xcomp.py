import logging
import time

from tensorlog import comline
from tensorlog import config
from tensorlog import comline
from tensorlog import declare
from tensorlog import funs
from tensorlog import ops

TRAINING_TARGET_VARNAME = '_target_y'

class AbstractCrossCompiler(object):

  """ Base class for tensorlog -> [theano|tensorflow|....] cross-compiler """

  def __init__(self,prog):
    # We need to create variables in different namespaces for
    # different instances of an OpSeqFunction, so that the variables
    # used to represent OpSeqFunction intermediate values don't clash.
    # namespaces are defined by integer ids, and we allocate a new one
    # for every distinct OpSeqFunction that gets compiled.
    self.nextNamespaceId = 0
    # holds output of compilation - subclasses should initialize this
    self.ws = Workspace(self)
    # pointer back to the program and matrixdb
    self.prog = prog
    self.db = prog.db
    # a constant used to put a little bit of weight on the 'null
    # entity'
    self.globalsInitialized = None
    logging.debug('AbstractCrossCompiler initialized %.3f Gb' % comline.memusage())

  def initGlobals(self):
    if not self.globalsInitialized:
      self.nullSmoothing = self.constantVector("_nullSmoothing",self.db.nullMatrix(1)*(1e-5))
    self.globalsInitialized = True

  def allocNamespacer(self):
    """Allocate a new NameSpacer object, which has its own unique namespace. """
    result = NameSpacer(self.nextNamespaceId)
    self.nextNamespaceId += 1
    return result

  #
  # these all define the interface to the database.  instead of
  # returning a constant matrix M, they will return a 'handle
  # expression', i.e., a target-language expression that evaluates to
  # that matrix at learning time.  In the simple cases, this is just
  # the name for a shared variable, but it could be an expression
  # based on that variable (eg its transpose)
  #

  def vector(self, matMode):
    """ Wraps a call to db.vector()
    """
    assert matMode.arity==1
    key = (matMode.getFunctor(),1)
    if not self.ws.hasHandleExpr(key):
      variable_name = "v__" + matMode.getFunctor()
      #TODO: should this be sparse? should vector just be diagonal matrices?
      val = self.wrapDBVector(self.db.vector(matMode)) #ignores all but functor for arity 1
      self.ws.insertHandleExpr(key, variable_name, val)
    return self.ws.getHandleExpr(key)

  def constantVector(self, variable_name, val):
    """ Wrap a call to db.onehot(), db.zeros(), etc.
    """
    key = (variable_name,0)
    if not self.ws.hasHandleExpr(key):
      wrapped_val = self.wrapDBVector(val)
      self.ws.insertHandleExpr(key, variable_name, wrapped_val)
    return self.ws.getHandleExpr(key)

  def matrix(self,matMode,transpose=False):
    """ Wraps a call to db.matrix()
    """
    # cache an expression for the un-transposed version of the matrix
    assert matMode.arity==2
    key = (matMode.getFunctor(),2)
    canonicalMode = declare.asMode( "%s(i,o)" % matMode.getFunctor())
    if not self.ws.hasHandleExpr(key):
      variable_name = "M__" + matMode.getFunctor()
      val = self.wrapDBMatrix(self.db.matrix(canonicalMode,False))
      self.ws.insertHandleExpr(key, variable_name, val)
    if self.db.transposeNeeded(matMode,transpose):
      return self.transposeMatrixExpr(self.ws.getHandleExpr(key))
    else:
      return self.ws.getHandleExpr(key)

  def ones(self):
    """Wraps a call to db.ones() """
    return self.constantVector('__ones',self.db.ones())

  def zeros(self):
    """Wraps a call to db.zeros() """
    return self.constantVector('__zeros',self.db.zeros())

  def onehot(self,sym):
    """Wraps a call to db.onehot() """
    return self.constantVector(sym,self.db.onehot(sym))

  #
  # compilation
  #

  def compile(self,funSpec,params=None):
    """Compile a tensorlog function to theano.  Params are optional, if
    they are given then also compile gradient of the loss function
    with respect to these parameters.  Params should be a list of
    (functor,arity) pairs, and funSpec should be a mode, a
    string encoding a mode, or a funs.Function
    """
    startTime = time.time()
    def status(msg): logging.info('%s time %.3f sec mem %.3f Gb' % (msg,time.time()-startTime,comline.memusage()))

    if isinstance(funSpec,declare.ModeDeclaration):
      status('tensorlog compiling %s' % funSpec)
      fun = self.prog.compile(funSpec)
    elif isinstance(funSpec,str):
      status('tensorlog compiling %s' % funSpec)
      fun = self.prog.compile(declare.asMode(funSpec))
    elif isinstance(funSpec,funs.Function):
      fun = funSpec
    else:
      assert False,'invalid function spec %r' % funSpec
    assert fun is not None
    status('tensorlog compilation complete')

    self.do_compile(fun,params)
    status('cross compilation complete')

  def do_compile(self,fun,params):
    self.initGlobals()
    # build the expression used for inference
    (self.ws.inferenceArgs,self.ws.inferenceExpr) = self.fun2Expr(fun)
    # do any postprocessing needed
    self.finalizeInference()
    # extend the inferenceExpr to also compute loss
    self.buildLossExpr(params)

  #
  # main compilation algorithm
  #

  def fun2Expr(self,fun,sharedInputs=None,depth=0):
    """Return a pair (inputs, expr) where binding the inputs in, and then
    evaluating the expression, is semantically equivalent to
    evaluating the Function fun in tensorlog, given that all the
    workspace variables are initialized.

    The sharedInputs is used if you already have created variables
    corresponding to the inputs to this expression.  This is the case
    when you have a SumFunction: all the subexpressions share the same
    inputs.

    Depth is the depth of recursion
    """

    if isinstance(fun,funs.SoftmaxFunction):
      inputs,subExpr = self.fun2Expr(fun.fun,sharedInputs,depth)
      return inputs,self.softmaxFun2Expr(subExpr)

    elif isinstance(fun,funs.SumFunction):
      assert(len(fun.funs)>=1)
      inputs,accum = self.fun2Expr(fun.funs[0],sharedInputs,depth)
      for f in fun.funs[1:]:
        (moreInputs,addend) = self.fun2Expr(f,inputs,depth)
        assert(len(moreInputs)==len(inputs))
        accum = self.addupExprs(accum,addend)
      return (inputs,accum)

    elif isinstance(fun,funs.OpSeqFunction):
      assert len(fun.opInputs)==1, 'mismatching number of inputs'
      # nspacer maps variables from the OpSeqFunction's environment to
      # subexpressions
      nspacer = self.allocNamespacer()
      seqInputs = []
      if sharedInputs==None:
        # create variables which will be used as inputs
        for v in fun.opInputs:
          nspacer[v] = self.createPlaceholder(nspacer.internalName(v),'vector')
          seqInputs.append(nspacer[v])
      else:
        # copy over the existing inputs to the new namespace
        assert len(fun.opInputs)==len(sharedInputs)
        for i in range(len(fun.opInputs)):
          v = fun.opInputs[i]
          nspacer[v] = sharedInputs[i]
          seqInputs.append(nspacer[v])
      # fill in the theano environment appropriately
      for op in fun.ops:
        nspacer[op.dst] = self.op2Expr(nspacer,op,depth)
      # return the inputs and the expression for the
      # OpSeqFunction's output
      return (seqInputs, nspacer[fun.ops[-1].dst])

    elif isinstance(fun,funs.NullFunction):
      return ([], self.zeros())

    else:
      assert False,'cannot cross-compile %r' % fun

  def op2Expr(self,nspacer,op,depth):
    """Extend a namespace with the expression for the output of the operation
    """
    if isinstance(op,ops.VecMatMulOp):
      return self.vecMatMulExpr(nspacer[op.src], self.matrix(op.matMode,op.transpose))
    elif isinstance(op,ops.AssignPreimageToVar):
      return self.vecMatMulExpr(self.ones(), self.matrix(op.matMode,True))
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return self.componentwiseMulExpr(nspacer[op.src], nspacer[op.src2])
    elif isinstance(op,ops.DefinedPredOp):
      _,subExpr = self.fun2Expr(op.subfun, [nspacer[op.src]], depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self.onehot(op.onehotConst)
    elif isinstance(op,ops.AssignVectorToVar):
      return self.vector(op.matMode)
    elif isinstance(op,ops.WeightedVec):
      return self.weightedVecExpr(nspacer[op.vec], nspacer[op.weighter])
    else:
      assert False,'cannot cross-compile %r' % op

  #
  # subclasses should implement these
  #

  # shared variables and placeholders

  def createPlaceholder(self,name,kind):
    """Create a placeholder for top-level inputs"""
    assert False, 'abstract method called'

  def insertHandleExpr(self,key,expr,val,kind):
    """Associate a DB object with the given key with an expression in the
    target language.  The key is a (functor,arity) pair.
    """
    assert False, 'abstract method called'

  # i/o

  def wrapMsg(self,vec):
    """ Convert a message/query vector in tensorlog's default
    representation (scipy sparse vector/matrix) to the target language
    """
    assert False, 'abstract method called'

  def wrapDBVector(self,vec):
    """ Convert a vector from the DB into athe target language """
    assert False, 'abstract method called'

  def wrapDBMatrix(self,mat):
    """Convert a matrix from the DB into the target language """
    assert False, 'abstract method called'

  def unwrapOutputs(self,targetLanguageOutputs):
    """ Convert a list of outputs produced by the target language to a
    tensorlog output (a scipy sparse vector/matrix)
    """
    return map(lambda v:self.unwrapOutput(v), targetLanguageOutputs)

  def unwrapOutput(self,targetLanguageOutputs):
    """ Convert an output produced by the target language to a
    tensorlog output (a scipy sparse vector/matrix)
    """
    assert False,'abstract method called'

  def unwrapUpdate(self,key,up):
    """ Convert updates for a parameter to generated by the target
    language to tensorlog's expected representation for updates.
    """
    assert False,'abstract method called'

  def unwrapParameterValue(self,key,val):
    """ Convert the value of learned parameter to tensorlog's format
    """
    assert False,'abstract method called'

  #
  # Primitive routines to produce target-language expressions from
  # smaller subexpressions.
  #

  # this works for most targets
  def addupExprs(self,accum,addend):
    """ Return an expression for the sum of two subexpressions.
    """
    return accum+addend

  def transposeMatrixExpr(self,m):
    """ Return the expression to transpose a matrix """
    assert False, 'abstract method called'

  def softmaxFun2Expr(self,fun):
    """ Return the expression to compute softmax of a function output """
    assert False, 'abstract method called'

  def vecMatMulExpr(self,v,m):
    """ Vector-matrix dot product """
    assert False, 'abstract method called'

  def componentwiseMulExpr(self,v1,v2):
    """ Component-wise multiplication """
    assert False, 'abstract method called'

  def weightedVecExpr(self,vec,weighter):
    """ Special operation used in tensorlog: component-wise multiply the
    vector with column sum of the weighter. """
    assert False, 'abstract method called'

  def finalizeInference(self):
    """ Any additional bookkeeping needed after inference expression is constructed. """
    assert False, 'abstract method called'

  def buildLossExpr(self,params):
   """Subroutine to extend the inference expression with all the stuff
   relating to loss. This is run after finalizeInference, so it
   assumes inference expression is constructed.  This also will
   compute any gradients that are needed. """
   assert False, 'abstract method called'

  def eval(self,rawInputs):
    """ Evaluate the inference expression.
    """
    assert False, 'abstract method called'

  def evalDataLoss(self,rawInputs,rawTarget):
    """Evaluate the unregularized loss of the data.  rawInputs will
    usually be [x,target_y] plus the parameters, and parameters are
    passed in in as (pred,arity) pairs.
    """
    assert False, 'abstract method called'

  def evalDataLossGrad(self,rawInputs,rawTarget):
    """Evaluate the gradient of the unregularized loss of the data.
    Inputs are the same as for evalDataLoss.
    """
    assert False, 'abstract method called'

  def exportAllLearnedParams(self):
    """Replace the parameter values in self.prog.db with the values that
    have been learned.
    """
    for key in self.ws.params:
      functor,arity = key
      newVal = self.getLearnedParam(key)
      self.db.setParameter(functor,arity,newVal)

  def getLearnedParam(self,key):
    """Replace the parameter values in self.prog.db with the value that
    was learned for this parameter.
    """
    assert False, 'abstract method called'

# some helper classes

class NameSpacer(object):
  """A 'namespaced' dictionary indexed by strings. Assigns every string
    to a string 'internalName' (which depends on the string and the
    namespaceId for this object) and indexes by that internal name.
  """

  def __init__(self,namespaceId):
    self.namespaceId = namespaceId
    self.env = {}
  def internalName(self,key):
    return 'n%d__%s' % (self.namespaceId,key)
  def __getitem__(self,key):
    return self.env[self.internalName(key)]
  def __setitem__(self,key,val):
    self.env[self.internalName(key)] = val

class Workspace(object):
  """ Holds information created in cross-compilation.
  """
  def __init__(self,xcomp):

    # backpointer to cross-compiler
    self.xcomp = xcomp

    # expression used for inference
    self.inferenceExpr = None
    # list of arguments to inferenceExpr - generated with createPlaceholder
    self.inferenceArgs = None
    # expression used for unregularized loss
    self.dataLossExpr = None
    # additional arguments for computing loss - generated with createPlaceholder
    self.dataLossArgs = None
    # gradient of loss expression wrt each parameter
    self.dataLossGradExprs = None

    # the workspace also stores 'handle expressions' for some of the
    # objects in the tensorlog database.  The DB objects are named, in
    # tensorlog, by a (functor,arity) pair. DB constants are given
    # arity zero.  Handle expressions should be inserted by
    # calling xcomp.insertHandleExpr()
    self._handleExpr = {}
    # there is some underlying variable with a gradient that is used
    # in every handle expression - sometimes this is the same as the
    # handle expression, but not always.  these are indexed by key
    self._handleExprVar = {}

    # parameters to optimize, as a list of keys
    self.params = []

  def hasHandleExpr(self,key):
    """ Check if a handle expression has been assigned to this key """
    return key in self._handleExpr

  def getHandleExpr(self,key):
    """return the handle expression for a matrixdb relation """
    return self._handleExpr[key]

  def getHandleExprVariable(self,key):
    """Return the variable whose gradient is used to adjust the expression
    for a matrixdb relation """
    return self._handleExprVar[key]

  def insertHandleExpr(self, key, varName, val):
    """Insert a new handle expression, by delegation to the containing
    cross-compiler """
    self.xcomp.insertHandleExpr(key,varName,val)

  def getParamVariables(self):
    """ Convenience method to find variables corresponding to paramaters """
    return map(lambda key:self.getHandleExprVariable(key), self.params)
