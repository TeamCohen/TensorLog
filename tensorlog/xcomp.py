# (C) William W. Cohen and Carnegie Mellon University, 2017

import logging
import time

from tensorlog import comline
from tensorlog import config
from tensorlog import declare
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import ops
from tensorlog import util

conf = config.Config()
conf.reparameterizeMatrices = True; conf.help.reparameterizeMatrices = 'pass parameter matrices through a softplus to make keep them positive'
conf.ignoreTypeCheck= False; conf.help.ignoreTypeCheck = 'allow unknown types in a database with types'

TRAINING_TARGET_VARNAME = '_target_y'

class AbstractCrossCompiler(object):

  """ Base class for tensorlog -> [theano|tensorflow|....] cross-compiler """

  def __init__(self,prog):
    # We need to create variables in different namespaces for
    # different instances of an OpSeqFunction, so that the variables
    # used to represent OpSeqFunction intermediate values don't clash.
    # namespaces are defined by integer ids, and we allocate a new one
    # for every distinct OpSeqFunction that gets compiled.
    self._nextNamespaceId = 0
    # holds outputs of compilation, indexed by mode
    self._wsDict = {}
    # holds output of current compilation process
    self.ws = None
    # pointers back to the program and matrixdb
    self.prog = prog
    self.db = prog.db
    # when a db is 'typeless', ie all entities are of type
    # matrixdb.THING, then _onlyType is set to THING
    self._onlyType = None
    # maps typeName to the vector used to introduce NULL entities,
    # with low weight, into a vector of type typeName
    self._nullSmoother = {}
    # set after vectors are allocated for the nullSmoother's
    self._globalsSet = None
    # Cache 'handle expressions' for some of the objects in the
    # tensorlog database.  The handle expressions are indexed by a
    # (functor,arity) pair. Handle expressions must be inserted by
    # calling insertHandleExpr().
    self._handleExpr = {}
    # For each handle expression, there is some underlying variable
    # with a gradient that is optimized.  Often this is the same as the
    # handle expression, but not always.  These are indexed by
    # functor,arity key.
    self._handleExprVar = {}
    logging.debug('AbstractCrossCompiler initialized %.3f Gb' % util.memusage())

  #
  # external UI
  #

  def close(self):
    """ Release any resources
    """
    pass

  def inference(self,mode,inputs=None):
    """ Returns (args,inferenceExpr) """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].inferenceArgs, self._wsDict[mode].inferenceExpr

  def inferenceFunction(self,mode,wrapInputs=True,unwrapOutputs=True):
    """Returns a python function which performs inference for the function
    defined by that mode.  The function takes a length-one tuple
    containing one argument X, which can be a row vector or a
    minibatch, and outputs a matrix with the same number of rows as X,
    and the number of columns appropriate for the output type of the
    mode.
    """
    args,expr = self.inference(mode)
    assert len(args)==1
    return self._asOneInputFunction(args[0],expr,wrapInputs,unwrapOutputs)

  def inferenceOutputType(self,mode,inputs=None):
    """ The type associated with the output of a tensorlog function.
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].tensorlogFun.outputType

  def proofCount(self,mode,inputs=None):
    """ Returns (args,proofCountExpr) """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].proofCountArgs, self._wsDict[mode].proofCountExpr

  def proofCountFunction(self,mode,wrapInputs=True,unwrapOutputs=True,inputs=None):
    """Returns a python function which performs counts proofs for the
    queries defined by that mode.  The function takes a length-one
    tuple containing one argument X, which can be a row vector or a
    minibatch, and outputs a matrix with the same number of rows as X,
    and the number of columns appropriate for the output type of the
    mode.
    """
    args,expr = self.proofCount(mode,inputs=inputs)
    assert len(args)==1
    return self._asOneInputFunction(args[0],expr,wrapInputs,unwrapOutputs)

  def proofCountOutputType(self,mode,inputs=None):
    """ The type associated with the output of a tensorlog function.
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].tensorlogFun.outputType

  def dataLoss(self,mode,inputs=None):
    """ Returns (args,dataLossExpr) """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].dataLossArgs, self._wsDict[mode].dataLossExpr

  def dataLossFunction(self,mode,wrapInputs=True,unwrapOutputs=True):
    """Returns a python function which compute the unregularized loss for
    the function defined by that mode, relative to target outputs Y.
    The function takes a single argument which is a list of (X,Y).
    """
    args,expr = self.dataLoss(mode)
    assert len(args)==2
    return self._asTwoInputFunction(args[0],args[1],expr,wrapInputs,unwrapOutputs)

  def dataLossGrad(self,mode,inputs=None):
    """Returns (args,[dataLossGrad1,....,dataLossGradn]), where each grad
    is the gradient of one of the parameters.The order of the grads
    is the same as the parameters.
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].dataLossArgs, self._wsDict[mode].dataLossGradExprs

  def dataLossGradFunction(self,mode,wrapInputs=True,unwrapOutputs=True,inputs=None):
    """Returns a python function which performs inference for the function
    defined by that mode.  The function takes a single argument which
    is a list of (X,Y).
    """
    args,exprList = self.dataLossGrad(mode,inputs=inputs)
    assert len(args)==2
    return self._exprListAsUpdateFunction(args[0],args[1],exprList,wrapInputs,unwrapOutputs)

  #
  # forwarded to the underlying database, or appropriate subclass
  # routine
  #

  def asSymbol(self,symbolId,typeName=None):
    """ Convert a typed integer id to a symbol
    """
    return self.db.asSymbol(symbolId,typeName=typeName)

  def asSymbolId(self,symbol,typeName=None):
    """ Convert a typed symbol to an integer id
    """
    return self.db.asSymbolId(symbol,typeName=typeName)

  def wrapInput(self,x):
    """ Convert scipy matrix to required input format
    """
    return self._wrapMsg(x)

  def unwrapInput(self,x):
    """Inverts wrapInput.  Override this only if inputs and outputs are
    in a different format.
    """
    return self._unwrapOutput(x)

  def unwrapOutput(self,y):
    """ Convert output to scipy matrix
    """
    return self._unwrapOutput(y)

  def unwrapParam(self,y):
    """ Convert output to scipy matrix
    """
    return self._unwrapOutput(x)


  def possibleOps(self,subExpr,typeName=None):
    """If a typeName is specified, then return a (expr,type) pairs, where
    each expression performs one primitive tensorlog operation on the
    subExpr given as input, and type is the name of the type for the
    resulting subExpr.

    If the typeName is NONE,

    """
    # TODO add multiple-input and zero-input operations
    if typeName is None:
      typeName = matrixdb.THING
      assert self.db.isTypeless(),'if database has types declared, you must specify the type of the input to possibleOps'
    result = []
    for (functor,arity) in self.db.matEncoding:
      if arity==2:
        mode = declare.asMode("%s(i,o)" % functor)
        if self.db.schema.getDomain(functor,arity)==typeName:
          op = self._vecMatMulExpr(subExpr, self._matrix(mode,transpose=False))
          if self.db.isTypeless():
            result.append(op)
          else:
            result.append((op,self.db.schema.getRange(functor,arity)))
        if self.db.schema.getRange(functor,arity)==typeName:
          op = self._vecMatMulExpr(subExpr, self._matrix(mode,transpose=True))
          if self.db.isTypeless():
            result.append(op)
          else:
            result.append((op,self.db.schema.getDomain(functor,arity)))
    return result

  #
  # used in inferenceFunction, dataLossFunction, etc
  #

  def _asOneInputFunction(self,arg1,expr,wrapInputs,unwrapOutputs):
    """Return a python function which implements the expression,
    optionally 'wrapping' the input and outputs.  If inputs are
    wrapped passing in scipy sparse matrices is ok.  If outputs are
    unwrapped then output will be scipy sparse matrices."""
    assert False,'abstract method called'

  def _asTwoInputFunction(self,arg1,arg2,expr,wrapInputs,unwrapOutputs):
    """Analogous to _asOneInputFunction but takes two inputs"""
    assert False,'abstract method called'

  def _exprListAsUpdateFunction(self,args,exprList,params,wrapInputs,unwrapOutputs):
    """Similar to _exprListAsUpdateFunction, but returns a python function
    which returns a list of pairs (key,update), mapping parameter
    'keys' -- i.e., functor,arity pairs -- to updates of those
    parameters.
    """
    assert False,'abstract method called'

  def getParamVariables(self,mode,inputs=None):
    """Find target-language variables that are optimized to set the DB
    parameters.  These are the variables that will be optimized in
    learning.  Eg, if a weight vector V is reparameterized by passing
    it through an softplus, this will be the underlying variable V0
    such that softplus(V0)=V.
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return map(lambda key:self._handleExprVar[key], self.prog.getParamList())

  def getParamHandles(self,mode,inputs=None):
    """Find target-language variables corresponding to DB parameters.
    These are the variables that store or compute the values that
    correspond most closely to the parameters. Eg, if a weight vector
    V is reparameterized by passing it through an softplus, this will
    be the variable V such that V=softplus(V0), where V0 is optimized
    in learning.
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return map(lambda key:self._handleExpr[key], self.prog.getParamList())

  def parameterFromDBToExpr(self,functor,arity):
    return self._handleExpr.get((functor,arity))

  def parameterFromDBToVariable(self,functor,arity):
    return self._handleExprVar.get((functor,arity))

  def pprint(self,mode,inputs=None):
    """Return list of lines in a pretty-print of the underlying, pre-compilation function.
    To actual display these, use something like
      print "\n".join(xcompiler.pprint("predict/io"))
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode].tensorlogFun.pprint()

  def getWorkspace(self,mode,inputs=None):
    """ Return the workspace associated with a mode
    """
    mode = self.ensureCompiled(mode,inputs=inputs)
    return self._wsDict[mode]

  def _getParamVariables(self):
    """ Convenience method to find variables corresponding to paramaters """
    return map(lambda key:self._handleExprVar[key], self.xcomp.prog.getParamList())

  #
  # these all define the interface to the database.  instead of
  # returning a constant matrix M, they will return a 'handle
  # expression', i.e., a target-language expression that evaluates to
  # that matrix at learning time.  In the simple cases, this is just
  # the name for a shared variable, but it could be an expression
  # based on that variable (eg its transpose)
  #

  def _vector(self, matMode):
    """ Wraps a call to db.vector()
    """
    assert matMode.arity==1
    key = (matMode.getFunctor(),1)
    if not key in self._handleExpr:
      assert (matMode.functor,1) in self.db.matEncoding, 'DB does not contain a value for %s' % str(matMode)
      variable_name = "v__" + matMode.getFunctor()
      val = self._wrapDBVector(self.db.vector(matMode)) #ignores all but functor for arity 1
      self._insertHandleExpr(key, variable_name, val, broadcast=True)
    return self._handleExpr[key]

  def _constantVector(self, variable_name, val):
    """ Wrap a call to db.onehot(), db.zeros(), etc.
    """
    key = (variable_name,0)
    if not key in self._handleExpr:
      wrapped_val = self._wrapDBVector(val)
      self._insertHandleExpr(key, variable_name, wrapped_val, broadcast=True)
    return self._handleExpr[key]

  def _matrix(self,matMode,transpose=False):
    """ Wraps a call to db.matrix()
    """
    # cache an expression for the un-transposed version of the matrix
    assert matMode.arity==2
    key = (matMode.getFunctor(),2)
    canonicalMode = declare.asMode( "%s(i,o)" % matMode.getFunctor())
    if not key in self._handleExpr:
      assert (matMode.functor,2) in self.db.matEncoding, 'DB does not contain a value for %s' % str(matMode)
      variable_name = "M__" + matMode.getFunctor()
      val = self._wrapDBMatrix(self.db.matrix(canonicalMode,False))
      self._insertHandleExpr(key, variable_name, val)
    if self.db.transposeNeeded(matMode,transpose):
      return self._transposeMatrixExpr(self._handleExpr[key])
    else:
      return self._handleExpr[key]

  def _ones(self,typeName):
    """Wraps a call to db.ones() """
    return self._constantVector('__ones',self.db.ones(typeName))

  def _zeros(self,typeName):
    """Wraps a call to db.zeros() """
    return self._constantVector('__zeros',self.db.zeros(numRows=1,typeName=typeName))

  def _onehot(self,sym,typeName):
    """Wraps a call to db.ones() """
    return self._constantVector(sym,self.db.onehot(sym,typeName))

  def _preimageOnesType(self,mode):
    """Wraps a call to db """
    return self.db.matrixPreimageOnesType(mode)

  #
  # compilation
  #

  def ensureCompiled(self,mode,inputs=None):
    """Compile a tensorlog function to target language, and cache the
    result.  Returns the canonical name of the mode (which can be a
    string produced by a declare.ModeDeclaration) that points to the
    compiled workspace.

    Inputs can be used to specify the input placeholders for the
    inference and loss functions.
    """

    if isinstance(mode,str): mode = declare.asMode(mode)
    assert isinstance(mode,declare.ModeDeclaration), 'invalid mode %r' % mode

    if mode not in self._wsDict:
      self.ws = self._wsDict[mode] = Workspace(self)
      startTime = time.time()
      def status(msg): logging.info('%s time %.3f sec mem %.3f Gb' % (msg,time.time()-startTime,util.memusage()))
      status('compiling %s'%str(mode))
      fun = self.ws.tensorlogFun = self.prog.compile(mode)
      status('tensorlog compilation complete; cross-compiling %s'%str(mode))
      self._doCompile(fun,mode,inputs)
      status('tensorlog->target language compilation complete')
    else:
      self.ws = self._wsDict[mode]
    return mode

  def _doCompile(self,fun,mode,inputs):
    """Main compilation method.  Can be overridden by subclasses
    """
    self._setupGlobals()
    # build the expression used for inference
    if isinstance(fun,funs.SoftmaxFunction):
      # proofCountExpr is the what we apply the softmax normalization to
      (self.ws.proofCountArgs,self.ws.proofCountExpr,self.ws.proofCountOutputType) = \
          self._fun2Expr(fun.fun,sharedInputs=inputs)
      self.ws.inferenceExpr = self._softmaxFun2Expr(self.ws.proofCountExpr,self.ws.proofCountOutputType)
      self.ws.inferenceArgs = self.ws.proofCountArgs
      self.ws.inferenceOutputType = self.ws.proofCountOutputType
    else:
      logging.warn('cannot recover proofCount expression for mode %s -  is it not softmax normalized?' % str(mode))
      (self.ws.inferenceArgs,self.ws.inferenceExpr,self.ws.inferenceOutputType) = \
          self._fun2Expr(fun,sharedInputs=inputs)
    # extend the inferenceExpr to also compute loss
    self._buildLossExpr(mode)
    self._finalizeCompile(mode)

  def _finalizeCompile(self,mode):
    """ Hook function called after _doCompile
    """
    pass

  def _setupGlobals(self):
    """ Initialize variables used by this cross-compiler object. """
    if not self._globalsSet:
      for typeName in self.db.schema.getTypes():
        self._nullSmoother[typeName] = self._constantVector("_nullSmoother_"+typeName,self.db.nullMatrix(numRows=1,typeName=typeName)*(1e-5))
      if self.db.isTypeless():
        self._onlyType = self.db.schema.defaultType()
      self._globalsSet = True

  #
  # recursive compilation of function tree
  #

  def _fun2Expr(self,fun,sharedInputs=None,depth=0):
    """Convert a tensorlog funs.Function() to an expression in the target
    language.  Return a triple (inputs, expr, typeName) where binding
    the inputs in, and then evaluating the expression, is semantically
    equivalent to evaluating the Function fun in tensorlog, given that
    all the workspace variables are initialized.  typeName is the
    outputType of the function.

    The sharedInputs is used if you already have created variables
    corresponding to the inputs to this expression.  This is the case
    when you have a SumFunction: all the subexpressions share the same
    inputs.

    Depth is the depth of recursion.
    """

    if isinstance(fun,funs.SoftmaxFunction):
      logging.debug('compiling: %sSoftmax'%(' '*depth))
      inputs,subExpr,outType = self._fun2Expr(fun.fun,sharedInputs,depth)
      return inputs,self._softmaxFun2Expr(subExpr,outType),outType

    elif isinstance(fun,funs.SumFunction):
      logging.debug('compiling: %sSum'%(' '*depth))
      assert(len(fun.funs)>=1)
      inputs,accum,outType = self._fun2Expr(fun.funs[0],sharedInputs,depth)
      for f in fun.funs[1:]:
        (moreInputs,addend,accumOutType) = self._fun2Expr(f,inputs,depth)
        assert accumOutType==outType,"inconsistent types %s vs %s in SumFunction" % (outType,accumOutType)
        assert(len(moreInputs)==len(inputs))
        accum = self._addupExprs(accum,addend)
      return (inputs,accum,outType)

    elif isinstance(fun,funs.OpSeqFunction):
      logging.debug('compiling: %sOpSeq'%(' '*depth))
      assert len(fun.opInputs)==1, 'mismatching number of inputs'
      # allocate a new nspacer, which maps variables from the
      # OpSeqFunction's environment to subexpressions
      nspacer = NameSpacer(self._nextNamespaceId)
      self._nextNamespaceId += 1
      seqInputs = []
      if sharedInputs==None:
        # create variables which will be used as inputs
        for v,typeName in zip(fun.opInputs,self._wrapInputTypes(fun)):
          if (not self.db.isTypeless()) and (typeName is None) and (not conf.ignoreTypeCheck):
            logging.error('unknown type trying to compile function %s # %s' % (fun.pprintSummary(),fun.pprintComment()))
            logging.error('unknown type for %s - set xcomp.conf.ignoreTypeCheck to allow' % nspacer.internalName(v))
            assert False
          nspacer[v] = self._createPlaceholder(nspacer.internalName(v),'vector',typeName)
          seqInputs.append(nspacer[v])
      else:
        # copy over the existing inputs to the new namespace
        assert len(fun.opInputs)==len(sharedInputs)
        for i in range(len(fun.opInputs)):
          v = fun.opInputs[i]
          nspacer[v] = sharedInputs[i]
          seqInputs.append(nspacer[v])
      # implement each op
      for op in fun.ops:
        nspacer[op.dst] = self._op2Expr(nspacer,op,depth)
      # return the inputs and the expression for the
      # OpSeqFunction's output
      return (seqInputs, nspacer[fun.opOutput], self._wrapOutputType(fun))

    elif isinstance(fun,funs.NullFunction):
      logging.debug('compiling: %sNull'%(' '*depth))
      typeName = self._wrapOutputType(fun)
      return ([], self._zeros(typeName), typeName)

    else:
      assert False,'cannot cross-compile %r' % fun

  def _op2Expr(self,nspacer,op,depth):
    """ Compute and return an expr for the output of the op
    """
    if isinstance(op,ops.VecMatMulOp):
      return self._vecMatMulExpr(nspacer[op.src], self._matrix(op.matMode,op.transpose))
    elif isinstance(op,ops.AssignPreimageToVar):
      return self._vecMatMulExpr(self._ones(self._preimageOnesType(op.matMode)), self._matrix(op.matMode,True))
    elif isinstance(op,ops.CallPlugin):
      pluginFun = self.prog.plugins.definition(op.mode)
      return apply(pluginFun,map(lambda s:nspacer[s], op.srcs))
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return self._componentwiseMulExpr(nspacer[op.src], nspacer[op.src2])
    elif isinstance(op,ops.DefinedPredOp):
      _,subExpr,subExprType = self._fun2Expr(op.subfun, [nspacer[op.src]], depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self._onehot(op.onehotConst,op.dstType)
    elif isinstance(op,ops.AssignVectorToVar):
      return self._vector(op.matMode)
    elif isinstance(op,ops.WeightedVec):
      return self._weightedVecExpr(nspacer[op.vec], nspacer[op.weighter])
    else:
      assert False,'cannot cross-compile %r' % op

  # If the db is typeless, then self.onlyType is set to be that type,
  # otherwise it is None.  If the db is stateless we want to ignore
  # a function's outputType and inputTypes and replace them with
  # onlyType

  def _wrapOutputType(self,fun):
    """Replace outputTypes with _onlyType for a typeless database.
    """
    return self._onlyType or fun.outputType

  def _wrapInputTypes(self,fun):
    """Replace list of inputTypes with list of onlyType for a typeless
    database.
    """
    return [self._onlyType]*len(fun.inputTypes) if self._onlyType else fun.inputTypes

  def _ensureWrapped(self,X,Y,wrapped):
    return (X,Y) if wrapped else (self._wrapMsg(X),self._wrapMsg(Y))

  def _ensureUnwrapped(self,X,Y,wrapped):
    return (X,Y) if not wrapped else (self._unwrapOutput(X),self._unwrapOutput(Y))
  #
  # subclasses should implement these
  #

  # shared variables and placeholders

  def _createPlaceholder(self,name,matOrVec,typeName):
    """Create a placeholder for top-level inputs"""
    assert False, 'abstract method called'

  def _insertHandleExpr(self,key,expr,val,broadcast=False):
    """Associate a DB object with the given functor,arity key (and value
    'val') with an expression in the target language.  The key is a
    (functor,arity) pair.  See comments for
    Workspace.insertHandleExpr.
    """
    assert False, 'abstract method called'

  # i/o

  def _wrapMsg(self,vec):
    """ Convert a message/query vector in tensorlog's default
    representation (scipy sparse vector/matrix) to the target language
    """
    assert False, 'abstract method called'

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into athe target language """
    assert False, 'abstract method called'

  def _wrapDBMatrix(self,mat):
    """Convert a matrix from the DB into the target language """
    assert False, 'abstract method called'

  def _unwrapDBVector(self,key,vec):
    """ Convert a vector from the target language into the format used by the tensorlog DB """
    assert False, 'abstract method called'

  def _unwrapDBMatrix(self,key,mat):
    """ Convert a matrix from the target language into the format used by the tensorlog DB """
    assert False, 'abstract method called'

  def _unwrapOutput(self,targetLanguageOutputs):
    """ Convert an output produced by the target language to a
    tensorlog output (a scipy sparse vector/matrix)
    """
    assert False,'abstract method called'

  def _unwrapUpdate(self,key,up):
    """ Convert updates for a parameter to generated by the target
    language to tensorlog's expected representation for updates.
    """
    assert False,'abstract method called'

  #
  # Primitive routines to produce target-language expressions from
  # smaller subexpressions.
  #

  # this works for most targets
  def _addupExprs(self,accum,addend):
    """ Return an expression for the sum of two subexpressions.
    """
    return accum+addend

  def _transposeMatrixExpr(self,m):
    """ Return the expression to transpose a matrix """
    assert False, 'abstract method called'

  def _softmaxFun2Expr(self,fun,typeName):
    """ Return the expression to compute softmax of a function output """
    assert False, 'abstract method called'

  def _vecMatMulExpr(self,v,m):
    """ Vector-matrix dot product """
    assert False, 'abstract method called'

  def _componentwiseMulExpr(self,v1,v2):
    """ Component-wise multiplication """
    assert False, 'abstract method called'

  def _weightedVecExpr(self,vec,weighter):
    """ Special operation used in tensorlog: component-wise multiply the
    vector with column sum of the weighter. """
    assert False, 'abstract method called'

  def _buildLossExpr(self,params):
   """Subroutine to extend the inference expression with all the stuff
   relating to loss. This is run after finalizeInference, so it
   assumes inference expression is constructed.  This also will
   compute any gradients that are needed. """
   assert False, 'abstract method called'

  def exportAllLearnedParams(self,session=None):
    """Replace the parameter values in self.prog.db with the values that
    have been learned.
    """
    for (functor,arity) in self.prog.getParamList():
      newVal = self.getLearnedParam((functor,arity),session)
      self.db.setParameter(functor,arity,newVal)

  def getLearnedParam(self,key,session=None):
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
    # tensorlog function that will be cross-compiled
    self.tensorlogFun = None
    # expression for proof counts for (output|input), with args and output type
    self.proofCountExpr = None
    self.proofCountArgs = None #list of placeholders
    self.proofCountOutputType = None
    # expression used for inference, with args and output type
    self.inferenceExpr = None
    self.inferenceArgs = None
    self.inferenceOutputType = None
    # expression used for unregularized loss, with args and output type
    self.dataLossExpr = None
    self.dataLossArgs = None
    self.dataLossGradExprs = None
