from tensorlog import config
from tensorlog import declare
from tensorlog import funs
from tensorlog import ops

TRAINING_TARGET_VARNAME = '_target_y'

class AbstractCrossCompiler(object):

  """ Base class for tensorlog -> [theano|tensorflow|....] cross-compiler """

  def __init__(self,db):
    # We need to create variables in different namespaces for
    # different instances of an OpSeqFunction, so that the variables
    # used to represent OpSeqFunction intermediate values don't clash.
    # namespaces are defined by integer ids, and we allocate a new one
    # for every distinct OpSeqFunction that gets compiled.
    self.nextNamespaceId = 0
    # holds output of compilation - subclasses should initialize this
    self.ws = Workspace(self)
    # pointer back to the matrixdb
    self.db = db
    # a constant used to put a little bit of weight on the 'null
    # entity'
    self.nullSmoothing = self.constantVector("_nullSmoothing",self.db.nullMatrix(1)*(1e-5))

  def allocNamespacer(self):
    """Allocate a new name space. """
    result = NameSpacer(self.nextNamespaceId)
    self.nextNamespaceId += 1
    return result

  def vector(self, matMode):
    """ Wraps a call to db.vector()
    """
    assert matMode.arity==1
    key = (matMode.getFunctor(),1)
    if (key) not in self.ws:
      variable_name = "v__" + matMode.getFunctor()
      val = self.wrapDBVector(self.db.vector(matMode)) #ignores all but functor for arity 1
      self.ws.insertVariable(key, variable_name, val, 'vector')
    return self.ws[key]

  def constantVector(self, variable_name, val):
    """ Used to wrap a call to db.onehot(), db.zeros(), etc.
    """
    key = (variable_name,0)
    if key not in self.ws:
      wrapped_val = self.wrapDBVector(val)
      self.ws.insertVariable(key, variable_name, wrapped_val, 'vector')
    return self.ws[key]

  def matrix(self,matMode,transpose=False):
    """ Wraps a call to db.matrix()
    """
    # cache an expression for the un-transposed version of the matrix
    assert matMode.arity==2
    key = (matMode.getFunctor(),2)
    canonicalMode = declare.asMode( "%s(i,o)" % matMode.getFunctor())
    if (key) not in self.ws:
      variable_name = "M__" + matMode.getFunctor()
      val = self.wrapDBMatrix(self.db.matrix(canonicalMode,False))
      self.ws.insertVariable(key, variable_name, val, 'matrix')
    if self.db.transposeNeeded(matMode,transpose):
      return self.transposeMatrixExpr(self.ws[key])
    else:
      return self.ws[key]

  def ones(self):
    """Wraps a call to db.ones() """
    return self.constantVector('__ones',self.db.ones())

  def zeros(self):
    """Wraps a call to db.zeros() """
    return self.constantVector('__zeros',self.db.zeros())

  def onehot(self,sym):
    """Wraps a call to db.onehot() """
    return self.constantVector(sym,self.db.onehot(sym))

  def wrapSymbols(self,inputSyms):
    """ Convert a list of symbols to a list of one-hot vectors that can be sent to eval"""
    return map(lambda sym:self.wrapDBVector(self.db.onehot(sym)), inputSyms)

  def compile(self,fun,params=None):
    """Compile a tensorlog function to theano.  Params are optional, if
    they are given then also compile gradient of the loss function
    with respect to these parameters.  Params should be a list of
    (functor,arity) pairs.
    """
    # build the expression used for inference
    (self.ws.inferenceArgs,self.ws.inferenceExpr) = self.fun2Expr(fun)
    # do any postprocessing needed to call it
    self.finalizeInference()
    #self.buildLossExpr(params)

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
      return self.softmaxFun2Expr(fun,sharedInputs,depth)

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

  # variables and i/o

  def createPlaceholder(self,name,kind):
    """Create a placeholder for top-level inputs"""
    assert False, 'abstract method called'

  def createSharedVar(self,name,val,kind):
    """Create a shared variable in the target language"""
    assert False, 'abstract method called'

  def wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    assert False, 'abstract method called'

  def wrapDBMatrix(self,mat):
    """Convert a matrix from the DB into a value used by the target
    language """
    assert False, 'abstract method called'

  def unwrapOutputs(self,targetLanguageOutputs):
    """ Convert outputs of an eval'd target language expression
    to what Tensorlog needs for its eval and test code """
    return map(lambda v:self.unwrapOutput(v), targetLanguageOutputs)

  def unwrapOutput(self,targetLanguageOutputs):
    assert False,'abstract method called'

  # manipulate expressions

  # this works for most targets
  def addupExprs(self,accum,addend):
    """ Return an expression for the sum of two subexpressions.
    """
    return accum+addend

  def transposeMatrixExpr(self,mx):
    """ Transpose a matrix """
    assert False, 'abstract method called'

  def softmaxFun2Expr(fun):
    """ Compute softmax of vector """
    assert False, 'abstract method called'

  def vecMatMulExpr(self,v,m):
    """ Vector-matrix dot product """
    assert False, 'abstract method called'

  def componentwiseMulExpr(self,v1,v2):
    """ Component-wise multiplication """
    assert False, 'abstract method called'

  def weightedVecExpr(self,vec,weighter):
    """ Special operation: component-wise multiply
    vector with column sum of the weighter. """
    assert False, 'abstract method called'

  def buildLossExpr(params):
   """ Add in the stuff relating to loss"""
   assert False, 'abstract method called'

  # clean up more

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
    pass

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
  """ Holds information created in cross-compilation
  """
  def __init__(self,xcomp):
    self.sharedVariable = {}
    self.sharedVariableInfoByName = {}
    self.sharedVariableInfoByKey = {}
    self.sharedVariableList = []
    self.inferenceExpr = None
    self.inferenceArgs = None
    self.dataLossExpr = None
    self.dataLossArgs = None
    self.dataLossGradExprs = None
    self.dataLossGradArgs = None
    self.xcomp = xcomp

  # overload 'in'
  def __contains__(self, key):
    return key in self.sharedVariableInfoByKey

  # overload 'foo[]'
  def __getitem__(self, key):
    return self.sharedVariable[key]

  def insertVariable(self, key, varName, val, kind):
    """ Add a shared variable to the workspace.
    """
    self.sharedVariableList.append(varName)
    self.sharedVariableInfoByName[varName] = (key,varName,val,kind)
    self.sharedVariableInfoByKey[key] = (key,varName,val,kind)
    self.sharedVariable[key] = self.xcomp.createSharedVar(varName,val,kind)
