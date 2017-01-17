import tensorlog
import funs
import ops
import matrixdb
import declare
import mutil
import config

import theano
import theano.tensor as TT
import theano.tensor.basic as TTB
import theano.tensor.nnet as TNN
import theano.sparse as TS
import theano.sparse.basic as TSB
import scipy.sparse as SS
import numpy as NP

class AbstractCrossCompiler(object):
  """ Base class for tensorlog -> theano cross-compiler """
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
    # expected values
    self.subexprCacheVarBindings = {}

    # pointer back to the matrixdb
    self.db = db
    #
    # stuff below is set by compile
    #
    # an expression implementing the tensorlog eval function
    self.expr = None
    # list of variables which are the input argument(s) to self.expr
    self.exprArgs = None
    # an expression implementing unregularized the loss function
    self.dataLossExpr = None
    # list of variables which are the input argument(s) to
    # self.dataLossExpr
    self.dataTargetArgs = None
    # functions for inference and unregularized loss
    self.inferenceFun = None

  def allocNamespace(self):
    """Allocate a new name space. """
    result = self.nameSpace
    self.nameSpace += 1
    return result

  #
  # theano-specific?
  #

  def vector(self,matMode):
    assert matMode.arity==1
    #db vector ignores mode
    if (matMode) not in self.subexprCache:
      u = "v__" + matMode.getFunctor()
      v = self.db.vector(matMode)
      self.subexprCache[matMode] = self.theanoSharedVec(v, name=u)
      self.subexprCacheVarBindings[self.subexprCache[matMode]] = v
    return self.subexprCache[matMode]

  def matrix(self,matMode,transpose=False):
    """ Wraps a call to db.matrix(), but will cache the results as a variable or expression.
    """
    # cache an expression for the un-transposed version of the matrix
    assert matMode.arity==2
    if (matMode) not in self.subexprCache:
      u = "M__" + matMode.getFunctor()
      m = self.db.matrix(matMode,False)
      self.subexprCache[matMode] = self.theanoSharedMat(m, name=u)
      self.subexprCacheVarBindings[self.subexprCache[matMode]] = m
    if transpose:
      return self.subexprCache[matMode].T
    else:
      return self.subexprCache[matMode]

  def constant(self,key,msg):
    if key not in self.subexprCache:
      self.subexprCache[key] = self.theanoSharedMsg(msg,name=key)
      self.subexprCacheVarBindings[self.subexprCache[key]] = msg
    return self.subexprCache[key]

  def show(self):
    """ print a summary to stdout """
    print 'exprArgs',self.exprArgs
    print 'expr',theano.pp(self.expr)
    print 'expr.sum()',theano.pp(self.expr.sum())
    print 'debug expr.sum()\n',theano.printing.debugprint(self.expr.sum())
    print 'subexpr cache:'
    for k,v in self.subexprCacheVarBindings.items():
      print ' |',k,v
    print 'fun\n',theano.pp(self.inferenceFun.maker.fgraph.outputs[0])
    print 'debug fun\n',theano.printing.debugprint(self.inferenceFun.maker.fgraph.outputs[0])

  def compile(self,fun):
    """ Compile a tensorlog function to theano """
    (self.exprArgs,self.expr) = self.fun2Expr(fun,None)
    #print 'self.args',self.exprArgs
    #print 'self.expr',theano.pp(self.expr)
    self.inferenceFun = theano.function(inputs=self.exprArgs, outputs=self.expr, mode='DebugMode')
    self.buildDataLossExpr()
    # for convenience
    return self

  def buildDataLossExpr(self):
    # add the unregularized loss function, which is cross-entropy
    target_y = self.theanoRowVar('_target_y')
    self.dataTargetArgs = [target_y]
    self.dataLossExpr = TNN.nnet.categorical_crossentropy(target_y, self.expr).mean()

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

###############################################################################
# implementation for dense messages, dense relation matrices
###############################################################################


class DenseMatDenseMsgCrossCompiler(AbstractCrossCompiler):
  """ Use theano's numpy wrappers for everything """

  def __init__(self,db):
    AbstractCrossCompiler.__init__(self,db)
    self.denseMsg = True
    self.denseMat = True
    # when messages are dense,
    # make sure the NULL value is small but bigger than zero,
    # which will be the default value
    # self.nullSmoothing = theano.shared(self.densifyMsg(self.db.nullMatrix(1)*1e-5), name="nullSmoothing")
    self.nullSmoothing = self.theanoSharedMsg(self.db.nullMatrix(1)*1e-5, name="nullSmoothing")

  # over-ride these to get different set of sparse/dense choices

  def densifyMat(self,m): return self._densify(m)
  def densifyMsg(self,v): return self._densify(v)
  def densifyVec(self,v): return self._densify(v)

  def sparsifyMat(self,m): return self._sparsify(m)
  def sparsifyVec(self,v): return self._sparsify(v)
  def sparsifyMsg(self,v): return self._sparsify(v)

  def _densify(self,x):
    return x.todense()
  def _sparsify(self,x):
    sx = SS.csr_matrix(x)
    sx.eliminate_zeros()
    return sx

  # over-ride these for different types of theano row variables
  def theanoSharedMat(self,val,name=None): return theano.shared(self.densifyMat(val), name=name)
  def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
  def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
  def theanoRowVar(self,name): return TT.drow(name)


  #
  # evaluate the function
  #

  def evalSymbols(self,inputSyms):
    assert len(inputSyms)==len(self.exprArgs)
    def sym2Vector(sym): return densifyMsg(self.db.onehot(sym))
    inputs = map(lambda sym:self.densifyMsg(self.db.onehot(sym)), inputSyms)
    return self.eval(inputs)

  def eval(self,inputs):
    formalArgs = inputs
    theanoResult = self.inferenceFun(*formalArgs)
    return map(lambda v:self.sparsifyMsg(v), theanoResult)

  def evalLoss(self,rawInputs,rawTarget,paramKeys):
    # the loss depends on the rawInputs, which will usually be
    # [x,target_y] and the parameters, which here are
    # passed in as (pred,arity) keys
    inputs = map(self.densifyMsg,rawInputs)
    target = self.densifyMsg(rawTarget)
    paramVars = map(self.getParamVar, paramKeys)
    # current value of parameters
    paramVals = map(lambda (pred,arity):self.db.matEncoding[(pred,arity)], paramKeys)
    def show(tag,x): print tag,'type',type(x),'val',x
    def showList(tag,xs): print tag,'types',map(type,xs)
    showList('inputs',inputs)
    show('target',target)
    showList('paramVars',paramVars)
    print 'lossExpr',theano.pp(self.dataLossExpr)
    formalArgs = inputs + [target] + paramVals
    lossFun = theano.function(inputs=(self.exprArgs + self.dataTargetArgs + paramVars), outputs=self.dataLossExpr)
    return lossFun(*formalArgs)


  #
  # the main compilation routines
  #

  def ones(self):
    """Return a theano expression that denotes an all-ones row vector """
    return self.constant('__ones',self.db.ones())

  def onehot(self,sym):
    """Return a theano expression that denotes the onehot row vector for a constant """
    return self.constant(sym,self.db.onehot(sym))

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
      return (inputs, TNN.nnet.softmax(subExpr) + self.nullSmoothing)

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
          thEnv[v] = self.theanoRowVar(thEnv.internalName(v))
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

  def __init__(self,db):
    DenseMatDenseMsgCrossCompiler.__init__(self,db)
    self.denseMat = False

  # over-ride these to keep sparse matrices
  def densifyMat(self,m): return m
  def sparsifyMat(self,m): return m

  # over-ride these for different types of theano row variables
  def theanoSharedMat(self,val,name=None): return theano.shared(self.densifyMat(val), name=name)
  def theanoSharedMsg(self,val,name=None): return theano.shared(self.densifyMsg(val), name=name)
  def theanoSharedVec(self,val,name=None): return theano.shared(self.densifyVec(val), name=name)
  def theanoRowVar(self,name): return TT.drow(name)

  # operator expressions for sparse matrices
  def op2Expr(self,thEnv,op,depth):
    if isinstance(op,ops.VecMatMulOp):
      mExpr = self.matrix(op.matMode,op.transpose)
      return TSB.structured_dot(thEnv[op.src],mExpr)
    elif isinstance(op,ops.AssignPreimageToVar):
      mExpr = self.matrix(op.matMode)
      # TODO: not sure why this simple expression doesn't work: TSB.dot(self.ones(), mExpr.transpose())
      # return TSB.dot(self.ones(), mExpr.transpose())
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
