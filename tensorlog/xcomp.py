from tensorlog import config

class AbstractCrossCompiler(object):
  """ Base class for tensorlog -> [theano|tensorflow|....] cross-compiler """
  def __init__(self,db):
    # namespaces are defined by integers, and we allocate a new one
    # for every distinct OpSeqFunction that gets compiled, so that the
    # variables used to represent OpSeqFunction intermediate values
    # don't clash.  Specifically, the namespace is passed into an
    # expression environment when it's created to 'salt' all
    # expression names in that environment.
    self.nameSpace = 0
    # portable replacement for Theano 'shared variables' and
    # tensorflow 'Variables' - parameters or constants that are reused
    # in multiple places.  these would include DB relation
    # matrices/vectors and constants.  Maps a key ("predicate",arity)
    # to a target-language subexpression that evaluates to that
    # subexpression
    self.subexprCache = {}
    # maps variables used in the expressions in the exprCache to their
    # expected values, and give then a canonical ordering
    self.subexprCacheVarBindings = {}
    self.subexprCacheVarList = []

    # pointer back to the matrixdb
    self.db = db
    #
    # stuff below is set by compile
    #
    # an expression implementing the tensorlog eval function
    self.expr = None
    # list of variables which are the input argument(s) to self.expr
    self.exprArgs = None
    # compiled function for inference, corresponding to self.expr
    self.inferenceFun = None

    # an expression implementing unregularized the loss function
    self.dataLossExpr = None
    # list of variables which are additional inputs to dataLoss
    self.dataTargetArgs = None
    self.dataLossFun = None
    # parameters we will optimize against
    self.params = None
    # gradient of dataLossFun wrt those params
    self.dataLossGradExprs = None #list of expressions
    self.dataLossGradFun = None   #function that returns multiple outputs

    self.dataLossGradParams = None
    # a constant used to put a little bit of weight on the 'null
    # entity'
    self.nullSmoothing = self.constantVector("_nullSmoothing",self.db.nullMatrix(1)*(1e-5))

  def allocNamespace(self):
    """Allocate a new name space. """
    result = self.nameSpace
    self.nameSpace += 1
    return result

  def getSubExpr(self,param):
    """ Find a subexpression/variable that corresponds to a DB functor,arity pair """
    assert len(param)==2 and (param[1]==1 or param[1]==2), 'parameter spec should be functor,arity pair'
    assert param in self.subexprCache,'key is %r subexprCache is %r' % (param,self.subexprCache)
    return self.subexprCache[param]

  def vector(self, matMode):
    """ Wraps a call to db.vector(), but will cache the results as a variable
    """
    assert matMode.arity==1
    key = (matMode.getFunctor(),1)
    if (key) not in self.subexprCache:
      variable_name = "v__" + matMode.getFunctor()
      val = self._wrapDBVector(self.db.vector(matMode)) #ignores all but functor for arity 1
      self._extendSubexprCache(key, self._vectorVar(variable_name), val)
    return self.subexprCache[key]

  def constantVector(self, variable_name, val):
    key = (variable_name,0)
    if key not in self.subexprCache:
      wrapped_val = self._wrapDBVector(val)
      self._extendSubexprCache(variable_name, self._vectorVar(variable_name), wrapped_val)
    return self.subexprCache[variable_name]

  def matrix(self,matMode,transpose=False,matrixTransposer=None):
    """ Wraps a call to db.matrix(), but will cache the results as a variable or expression.
    """
    # cache an expression for the un-transposed version of the matrix
    assert matMode.arity==2
    key = (matMode.getFunctor(),2)
    if (key) not in self.subexprCache:
      variable_name = "M__" + matMode.getFunctor()
      val = self._wrapDBMatrix(self.db.matrix(matMode,False))
      self._extendSubexprCache(key, self._matrixVar(variable_name), val)
    if self.db.transposeNeeded(matMode,transpose):
      return matrixTransposer(self.subexprCache[key])
    else:
      return self.subexprCache[key]

  def ones(self):
    """Wraps a call to db.ones(), but will cache the result """
    return self.constantVector('__ones',self.db.ones())

  def zeros(self):
    """Wraps a call to db.zeros(), but will cache the result """
    return self.constantVector('__zeros',self.db.zeros())

  def onehot(self,sym):
    """Wraps a call to db.onehot(), but will cache the result """
    return self.constantVector(sym,self.db.onehot(sym))

  def _extendSubexprCache(self, key, var, val):
    self.subexprCache[key] = var
    self.subexprCacheVarBindings[var] = val
    self.subexprCacheVarList.append(var)

  def _secondaryArgs(self):
    return self.subexprCacheVarList

  def _secondaryArgBindings(self):
    return map(lambda v:self.subexprCacheVarBindings[v], self.subexprCacheVarList)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    assert False, 'abstract method called'

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    assert False, 'abstract method called'

  def _sparsify(self,msg):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    assert False, 'abstract method called'

  def _vectorVar(self,name):
    """Create a variable in the target language"""
    assert False, 'abstract method called'

  def _matrixVar(self,name):
    """Create a variable in the target language"""
    assert False, 'abstract method called'

  def compile(self,fun,params=None):
    """Compile a tensorlog function to theano.  Params are optional, if
    they are given then also compile gradient of the loss function
    with respect to these parameters.  Params should be a list of
    (functor,arity) pairs.
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
