import theano
import theano.tensor as TT
import theano.tensor.basic as TTB
import theano.tensor.nnet as TNN
import theano.sparse as TS
import theano.sparse.basic as TSB
import theano.sparse.type as TST
import scipy.sparse as SS
import numpy as NP

import config
import funs
import ops

class AbstractCrossCompiler(object):
  """ Base class for tensorlog -> theano cross-compiler
  """
  def __init__(self,db):
    # namespaces are defined by integers, and we allocate a new one
    # for every distinct OpSeqFunction that gets compiled, so that the
    # variables used to represent OpSeqFunction intermediate values
    # don't clash.  Specifically, the namespace is passed into an
    # expression environment when it's created to 'salt' all
    # expression names in that environment.
    self.nameSpace = 0
    # portable replacement for 'shared variables' - parameters or
    # constants that are reused in multiple places.  these would
    # include DB relation matrices/vectors and constants.
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

  def matrix(self,matMode,transpose=False):
    """ Wraps a call to db.matrix(), but will cache the results as a variable or expression.
    """
    # cache an expression for the un-transposed version of the matrix
    print '==> called matrix',matMode,transpose
    assert matMode.arity==2
    key = (matMode.getFunctor(),2)
    if (key) not in self.subexprCache:
      variable_name = "M__" + matMode.getFunctor()
      val = self._wrapDBMatrix(self.db.matrix(matMode,False))
      self._extendSubexprCache(key, self._matrixVar(variable_name), val)
    if self.db.transposeNeeded(matMode,transpose):
      return self.subexprCache[key].T
    else:
      return self.subexprCache[key]

  def ones(self):
    """Wraps a call to db.ones(), but will cache the result """
    return self.constantVector('__ones',self.db.ones())

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


class TheanoCrossCompiler(AbstractCrossCompiler):

  def compile(self,fun,params=None):
    """Compile a tensorlog function to theano.  Params are optional, if
    they are given then also compile gradient of the loss function.
    Params should be a list of (functor,arity) pairs.
    """
    (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
    self.inferenceFun = theano.function(inputs=(self.exprArgs + self._secondaryArgs()),
                                        outputs=self.expr,
                                        mode='DebugMode')
    self._buildLossExpr(params)
    return self

  def _buildLossExpr(self,params):
    """ Add in the stuff relating to loss"""
    target_y = self._vectorVar('_target_y')
    self.dataTargetArgs = [target_y]
    #this expr is safe for vectors with zeros - TNN.nnet.categorical_crossentropy blows up
    #self.dataLossExpr = (-target_y*TT.log(self.expr+1)).mean()
    # get the non-zero values of the dense expression
    self.dataLossExpr = (target_y*self.applyOpToNonzerosOfDense(TT.log,self.expr)).mean()
    self.dataLossFun = theano.function(
        inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
        outputs=(self.expr,self.dataLossExpr),
        mode='DebugMode')
    if params is not None:
      self.params = params
      paramVars = map(self.getSubExpr,params)
      self.dataLossGradExprs = theano.grad(self.dataLossExpr, paramVars)
      self.dataLossGradFun = theano.function(
          inputs=(self.exprArgs + self.dataTargetArgs + self._secondaryArgs()),
          outputs=self.dataLossGradExprs,
          mode='DebugMode')

  #
  # evaluators
  #

  def wrapSymbols(self,inputSyms):
    """ Convert a list of symbols to a list of one-hot vectors that can be sent to eval"""
    return map(lambda sym:self._wrapDBVector(self.db.onehot(sym)), inputSyms)

  def _unwrapOutputs(self,targetFunctionOutputs):
    return map(lambda v:self._sparsify(v), targetFunctionOutputs)

  def eval(self,inputs):
    formalArgs = inputs +self._secondaryArgBindings()
    return self._unwrapOutputs(self.inferenceFun(*formalArgs))

  def evalDataLoss(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self._wrapDBVector, rawInputs)
    target = self._wrapDBVector(rawTarget)
    formalArgs = inputs + [target] + self._secondaryArgBindings()
    return self._unwrapOutputs(self.dataLossFun(*formalArgs))

  def evalDataLossGrad(self,rawInputs,rawTarget):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self._wrapDBVector, rawInputs)
    target = self._wrapDBVector(rawTarget)
    formalArgs = inputs + [target] + self._secondaryArgBindings()
    return self._unwrapOutputs(self.dataLossGradFun(*formalArgs))

  def applyOpToNonzerosOfDense(self,op,expr):
    sparseExpr = TSB.clean(TSB.csr_from_dense(expr))
    newData = op(TSB.csm_data(sparseExpr)).flatten()
    newSparse = TS.CSR(newData, TSB.csm_indices(sparseExpr), TSB.csm_indptr(sparseExpr), TSB.csm_shape(sparseExpr))
    return TSB.dense_from_sparse(newSparse)

  # for debugging output

  def show(self):
    """ print a summary to stdout """
    print 'exprArgs',self.exprArgs
    print 'expr',theano.pp(self.expr)
    print 'expr.sum()',theano.pp(self.expr.sum())
    #print 'debug expr.sum()\n',theano.printing.debugprint(self.expr.sum())
    print 'subexpr cache:'
    for k,v in self.subexprCacheVarBindings.items():
      print ' |',k,'type',type(v)
    print 'function:',theano.pp(self.inferenceFun.maker.fgraph.outputs[0])
    #print 'debug fun\n',theano.printing.debugprint(self.inferenceFun.maker.fgraph.outputs[0])


  def debugExpr(self):
    AbstractCrossCompiler.debugVar(self.expr,0,maxdepth=20)

  @staticmethod
  def debugVar(v,depth=0,maxdepth=10):
    if depth>maxdepth:
      print '...'
    else:
      print '| '*(depth+1),
      print 'var: name',v.name,'type',type(v),'def',theano.pp(v)
      for a in v.get_parents():
        AbstractCrossCompiler.debugApply(a,depth=depth+1,maxdepth=maxdepth)

  @staticmethod
  def debugApply(a,depth=0,maxdepth=10):
    if depth>maxdepth:
      print '...'
    else:
      print '| '*(depth+1),
      print 'apply: ',a,'op',type(a.op),'output types',map(type,a.outputs)
      for v in a.inputs:
        AbstractCrossCompiler.debugVar(v,depth=depth+1,maxdepth=maxdepth)

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################


class DenseMatDenseMsgCrossCompiler(TheanoCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def _vectorVar(self,name):
    return TT.drow(name)

  def _matrixVar(self,name):
    return TT.dmatrix(name)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat.todense()

  def _sparsify(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx


    #
    # the main compilation routines
    # 

  def fun2Expr(self,fun,sharedInputs=None,depth=0):
      """Return a pair (inputs, expr) where binding the inputs in theano,
      and then evaluating the expression, is roughly equivalent to
      evaluating the Function fun in tensorlog.  It's only roughly
      equivalent because one also needs to bind the necessary
      variables from the matrixdb to their values.
  
      The sharedInputs is used if you already have theano variables
      corresponding to the inputs to this expression.  This is the case
      when you have a SumFunction: all the subexpressions share the same inputs.
      """
  
      if isinstance(fun,funs.SoftmaxFunction):
        # wrap inner function with softmax function
        inputs,subExpr = self.fun2Expr(fun.fun,sharedInputs,depth=depth)
        softmaxOverNonzeros = self.applyOpToNonzerosOfDense(TNN.nnet.softmax,subExpr+self.nullSmoothing)
        return (inputs, softmaxOverNonzeros)
  
      elif isinstance(fun,funs.SumFunction):
        assert(len(fun.funs)>=1)
        inputs,accum = self.fun2Expr(fun.funs[0],sharedInputs,depth=depth)
        for f in fun.funs[1:]:
          (moreInputs,addend) = self.fun2Expr(f,inputs,depth=depth)
          assert(len(moreInputs)==len(inputs))
          accum = accum+addend
        return (inputs,accum)
  
      elif isinstance(fun,funs.OpSeqFunction):
        assert len(fun.opInputs)==1, 'mismatching number of inputs'
        # thEnv, a 'theano environment', maps nameSpaced variables
        # from the OpSeqFunction's environment to the
        # corresponding theano subexpressions
        thEnv = NameSpacer(self.allocNamespace())
        seqInputs = []
        if sharedInputs==None:
          # create the list of theano variables, which should be
          # used as inputs to the expression
          for v in fun.opInputs:
            thEnv[v] = self._vectorVar(thEnv.internalName(v))
            seqInputs.append(thEnv[v])
        else:
          # copy over the existing inputs to the new environment
          assert len(fun.opInputs)==len(sharedInputs)
          for i in range(len(fun.opInputs)):
            v = fun.opInputs[i]
            thEnv[v] = sharedInputs[i]
            seqInputs.append(thEnv[v])
        # fill in the theano environment appropriately
        for op in fun.ops:
          thEnv[op.dst] = self.op2Expr(thEnv,op,depth)
        # return the inputs and the expression for the
        # OpSeqFunction's output
        return (seqInputs, thEnv[fun.ops[-1].dst])
  
      else:
        assert False,'cannot cross-compile %r' % fun
  
  # operator expressions for dense matrices
  def op2Expr(self,thEnv,op,depth):
      """Extend the theano environment with an expression for the
      destination of the Operator, for dense matrices
      """
      if isinstance(op,ops.VecMatMulOp):
        mExpr = self.matrix(op.matMode,op.transpose)
        return thEnv[op.src].dot(mExpr)
      elif isinstance(op,ops.AssignPreimageToVar):
        mExpr = self.matrix(op.matMode)
        return self.ones().dot(mExpr.T)
      elif isinstance(op,ops.ComponentwiseVecMulOp):
        return thEnv[op.src] * thEnv[op.src2]
      elif isinstance(op,ops.DefinedPredOp):
        _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
        return subExpr
      elif isinstance(op,ops.AssignOnehotToVar):
        return self.onehot(op.onehotConst)
      elif isinstance(op,ops.AssignVectorToVar):
        return self.vector(op.matMode)
      elif isinstance(op,ops.WeightedVec):
        return thEnv[op.vec] * TT.sum(thEnv[op.weighter], axis=1, keepdims=True)
      else:
        assert False,'cannot cross-compile %r' % op

###############################################################################
# implementation for dense messages, sparse relation matrices
###############################################################################

class SparseMatDenseMsgCrossCompiler(DenseMatDenseMsgCrossCompiler):
  """ Use theano's numpy wrappers for everything
  """

  def _vectorVar(self,name):
    return TT.drow(name)

  def _matrixVar(self,name):
    return TSB.matrix('csr',name=name)

  def _wrapDBVector(self,vec):
    """ Convert a vector from the DB into a vector value used by the
    target language """
    return vec.todense()

  def _wrapDBMatrix(self,mat):
    """ Convert a matrix from the DB into a vector value used by the
    target language """
    return mat

  def _sparsify(self,x):
    """Convert a matrix produced by the target language to the usual
    sparse-vector output of tensorlog"""
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  # operator expressions for sparse matrices
  def op2Expr(self,thEnv,op,depth):
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose)
      return TSB.structured_dot(thEnv[op.src],mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode)
      # TODO: not sure why this simple expression doesn't work: TSB.dot(self.ones(), mExpr.transpose())
      return TSB.structured_dot(mExpr,self.ones().transpose()).transpose()
    elif isinstance(op,ops.ComponentwiseVecMulOp):
      return thEnv[op.src] * thEnv[op.src2]
    elif isinstance(op,ops.DefinedPredOp):
      _inputs,subExpr = self.fun2Expr(op.subfun,[thEnv[op.src]],depth=depth+1)
      return subExpr
    elif isinstance(op,ops.AssignOnehotToVar):
      return self.onehot(op.onehotConst)
    elif isinstance(op,ops.AssignVectorToVar):
      return self.vector(op.matMode)
    elif isinstance(op,ops.WeightedVec):
      return thEnv[op.vec] * TT.sum(thEnv[op.weighter], axis=1, keepdims=True)
    else:
      assert False,'cannot cross-compile %r' % op
