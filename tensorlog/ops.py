# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# operators - primitive actions that are performed in sequence during
#       function evaluation
#

import logging
import scipy.sparse

from tensorlog import opfunutil
from tensorlog import mutil
from tensorlog import config
import copy

conf = config.Config()
conf.trace = False;    conf.help.trace =       "Print debug info during op execution"
conf.long_trace = 0;   conf.help.long_trace =    "Print output of messages with < n nonzeros - only for small tasks"
conf.max_trace = False;  conf.help.max_trace =     "Print max value of functions after op"
conf.check_nan = True;   conf.help.check_overflow =  "Check if output of each op is nan."
conf.pprintMaxdepth=0;   conf.help.pprintMaxdepth =  "Controls op.pprint() output"


class Op(opfunutil.OperatorOrFunction):
  """Sort of like a function but side-effects an environment.  More
  specifically, this is the tensorlog encoding for matrix-db
  'operations' which can be 'eval'ed or differentiated. Operations
  typically specify src and dst variable names and eval-ing them
  will side-effect an environment, by binding the dst to some
  function of the input (src) bindings.
  """

  def __init__(self,dst):
    self.dst = dst
    # used for generating typed expressions
    self.dstType = None
    # used for debugging
    self.msgFrom = self.msgTo = None

  def setMessage(self,msgFrom,msgTo):
    """For debugging/tracing, record the BP message associated with this
    operation."""
    self.msgFrom = msgFrom
    self.msgTo = msgTo

  def eval(self,env,pad):
    """Evaluate an operator inside an environment."""
    if conf.trace:
      print 'op eval',self,
    self._doEval(env,pad)
    pad[self.id].output = env[self.dst]
    if conf.trace:
      def do_trace(foo):
        print 'stores',mutil.summary(foo),
        if conf.long_trace>foo.nnz: print 'holding',env.db.matrixAsSymbolDict(foo),
        if conf.max_trace: print 'max',mutil.maxValue(foo),
      do_trace(env[self.dst])
      if hasattr(self, 'sk'):
        do_trace(self.sk.unsketch(env[self.dst]))
      print
    if conf.check_nan:
      mutil.checkNoNANs(env[self.dst], context='saving %s' % self.dst)

  def backprop(self,env,gradAccum,pad):
    """Backpropagate errors - stored in the env.delta[...] from outputs of
    the operator to the inputs.  Assumes that 'eval' has been
    called first.
    """
    if conf.trace:
      print 'call op bp',self,'delta[',self.dst,'] shape',env.delta[self.dst].get_shape(),
      if conf.long_trace: print env.db.matrixAsSymbolDict(env.delta[self.dst])
      else: print
    self._doBackprop(env,gradAccum,pad)
    pad[self.id].delta = env.delta[self.dst]
    if conf.trace:
      print 'end op bp',self

  def pprint(self,depth=0):
    description = ('%-2d ' % self.id) + self.pprintSummary()
    comment = self.pprintComment()
    if comment: return [description + ' # ' + comment]
    else: return [description]

  def pprintSummary(self):
    rhs = self.dst if (self.dstType is None) else '%s(%s)' % (self.dst,self.dstType)
    return '%s = %s' % (rhs,self._ppLHS())

  def pprintComment(self):
    return '%s -> %s' % (self.msgFrom,self.msgTo) if (self.msgFrom and self.msgTo) else ''

  def _ppLHS(self):
    #override in subclasses
    return repr(self)

  #needed for traversal
  def children(self):
    #override in subclass
    return []

  def install(self,nextId):
    """ Give a numeric id to this operator/function """
    self.id = nextId
    return nextId+1

  def copy(self):
    assert False, "abstract method called"

class DefinedPredOp(Op):
  """Op that calls a defined predicate."""
  def __init__(self,tensorlogProg,dst,src,mode,depth):
    super(DefinedPredOp,self).__init__(dst)
    self.tensorlogProg = tensorlogProg
    self.src = src
    self.funMode = mode
    self.depth = depth
    #self.subfun = copy.deepcopy(self.tensorlogProg.function[(self.funMode,self.depth)])
    self.subfun = self.tensorlogProg.function[(self.funMode,self.depth)]
    self.dstType = self.subfun.outputType
  def __repr__(self):
    return "DefinedPredOp(%r,%r,%s,%d)" % (self.dst,self.src,str(self.funMode),self.depth)
  def _ppLHS(self):
    return "f_[%s,%d](%s)" % (str(self.funMode),self.depth,self.src)
  def _doEval(self,env,pad):
    vals = [env[self.src]]
    outputs = self.subfun.eval(self.tensorlogProg.db, vals, pad)
    env[self.dst] = outputs
  def _doBackprop(self,env,gradAccum,pad):
    newDelta = self.subfun.backprop(env.delta[self.dst],gradAccum,pad)
    env.delta[self.src] = newDelta
  def pprint(self,depth=-1):
    top = super(DefinedPredOp,self).pprint(depth)
    # depth here is depth of the recursion from DefinedPredOp's to Functions
    if depth>conf.pprintMaxdepth: return top + ["%s..." % ('| '*(depth+1))]
    else: return top + self.subfun.pprint(depth=depth+1)
  def install(self,nextId):
    """ Give a numeric id to this operator """
    self.id = nextId
    # only use deep copy if we have a duplicate
    if hasattr(self.subfun,'id'):
      self.subfun = self.subfun.copy() #copy.deepcopy(self.subfun)
      # NB copy.id is not set
    return self.subfun.install(nextId+1)
  def copy(self):
    return DefinedPredOp(self.tensorlogProg,self.dst,self.src,self.funMode,self.depth)
  def children(self):
    return [self.subfun]

class AssignPreimageToVar(Op):
  """Mat is something like p(X,Y) where Y is not used 'downstream' or
  p(X,c) where c is a constant.  Assign a row vector which encodes
  the preimage of the function defined by X to the environment
  variable 'dst'. """
  def __init__(self,dst,matMode,dstType=None):
    super(AssignPreimageToVar,self).__init__(dst)
    self.matMode = matMode
    if dstType is not None: self.dstType = dstType
  def __repr__(self):
    return "AssignPreimageToVar(%s,%s)" % (self.dst,self.matMode)
  def _ppLHS(self):
    return "M_[%s]" % str(self.matMode)
  def _doEval(self,env,pad):
    env[self.dst] = env.db.matrixPreimage(self.matMode)
  def _doBackprop(self,env,gradAccum,pad):
    #TODO implement preimages
    assert False,'backprop with preimages not implemented'
  def copy(self):
    return AssignPreimageToVar(self.dst,self.matMode)

class AssignVectorToVar(Op):
  """Mat is a unary predicate like p(X). Assign a row vector which
  encodes p to the variable 'dst'. """
  def __init__(self,dst,matMode,dstType=None):
    super(AssignVectorToVar,self).__init__(dst)
    self.matMode = matMode
    if dstType is not None: self.dstType = dstType
  def __repr__(self):
    return "AssignVectorToVar(%s,%s)" % (self.dst,self.matMode)
  def _ppLHS(self):
    return "V_[%s]" % str(self.matMode)
  def _doEval(self,env,pad):
    env[self.dst] = env.db.vector(self.matMode)
  def _doBackprop(self,env,gradAccum,pad):
    if env.db.isParameter(self.matMode):
      update = env.delta[self.dst]
      key = (self.matMode.functor,self.matMode.arity)
      gradAccum.accum(key,update)
  def copy(self):
    return AssignVectorToVar(self.dst,self.matMode)


class AssignOnehotToVar(Op):
  """Assign a one-hot row encoding of a constant to the dst variable.
  Mode is either assign(var,const) or assign(var,const,type)
  """
  def __init__(self,dst,mode):
    super(AssignOnehotToVar,self).__init__(dst)
    self.mode = mode
    assert self.mode.isConst(1),'second argument of assign/2 must be a constant'
    self.onehotConst = mode.arg(1)
    self.dstType = None
    if self.mode.getArity()==3:
      self.dstType = mode.arg(1)
  def __repr__(self):
    return "AssignOnehotToVar(%s,%s)" % (self.dst,self.onehotConst)
  def _ppLHS(self):
    return 'U_[%s]' % self.onehotConst
  def _doEval(self,env,pad):
    env[self.dst] = env.db.onehot(self.onehotConst,self.dstType)
  def _doBackprop(self,env,gradAccum,pad):
    pass
  def copy(self):
    return AssignOnehotToVar(self.dst,self.mode)

class VecMatMulOp(Op):
  """Op of the form "dst = src*mat or dst=src*mat.tranpose()"
  """
  def __init__(self,dst,src,matMode,transpose=False):
    super(VecMatMulOp,self).__init__(dst)
    self.src = src
    self.matMode = matMode
    self.transpose = transpose
  def __repr__(self):
    return "VecMatMulOp(%r,%r,%s,%r)" % (self.dst,self.src,self.matMode,self.transpose)
  def _ppLHS(self):
    buf = "%s * M_[%s]" % (self.src,self.matMode)
    if self.transpose: buf += ".T"
    return buf
  def _doEval(self,env,pad):
    env[self.dst] = env[self.src] * env.db.matrix(self.matMode,self.transpose)
  def _doBackprop(self,env,gradAccum,pad):
    # dst = f(src,mat)
    env.delta[self.src] = env.delta[self.dst] * env.db.matrix(self.matMode,(not self.transpose))
    mutil.checkCSR(env.delta[self.src],'delta[%s]' % self.src)
    if env.db.isParameter(self.matMode):
      update = env[self.src].transpose() * (env.delta[self.dst])
      update = scipy.sparse.csr_matrix(update)
      # The transpose flag is set in BP when sending a message
      # 'backward' from a goal output to variable, and indicates
      # if the operation needs to transpose the matrix.  Since
      # the db stores predicates p(a,b) internally as a matrix
      # where a is a row and b is a column, when the matMode is
      # p(o,i) then another internal transposition happens, by
      # the database.  We need to transpose the update when
      # exactly one of these transpositions happen, not two or
      # zero
      transposeUpdate = env.db.transposeNeeded(self.matMode,self.transpose)
      if transposeUpdate:
        update = update.transpose()
        update = scipy.sparse.csr_matrix(update)
      # finally save the update
      key = (self.matMode.functor,self.matMode.arity)
      mutil.checkCSR(update,'update for %s mode %s transpose %s' % (str(key),str(self.matMode),transposeUpdate))
      gradAccum.accum(key,update)
  def copy(self):
    return VecMatMulOp(self.dst,self.src,self.matMode,self.transpose)

class CallPlugin(Op):
  """Call out to a user-defined predicate.  These are currently only
  supported in cross-compilation.
  """
  def __init__(self,dst,srcs,mode,dstType=None):
    super(CallPlugin,self).__init__(dst)
    self.srcs = srcs
    self.mode = mode
    self.dstType = dstType
  def __repr__(self):
    return "BuiltInOp(%r,%r,%s)" % (self.dst,",".join(self.srcs),self.mode)
  def _ppLHS(self):
    return "CallPlugin{%s}(%s)" % (str(self.mode),",".join(self.srcs))
  def _doEval(self,env,pad):
    assert False,'CallPlugin only supported in cross-compilation'
  def _doBackprop(self,env,gradAccum,pad):
    assert False,'CallPlugin only supported in cross-compilation'
  def copy(self):
    return CallPlugin(self.dst,self.srcs,self.mode)

class ComponentwiseVecMulOp(Op):
  """ Computes dst = src*Diag(src2), i.e., the component-wise product of
  two row vectors.
  """
  def __init__(self,dst,src,src2):
    super(ComponentwiseVecMulOp,self).__init__(dst)
    self.src = src
    self.src2 = src2
  def __repr__(self):
    return "ComponentwiseVecMulOp(%r,%r,%s)" % (self.dst,self.src,self.src2)
  def _ppLHS(self):
    return "%s o %s" % (self.src,self.src2)
  def _doEval(self,env,pad):
    env[self.dst] = mutil.broadcastAndComponentwiseMultiply(env[self.src],env[self.src2])
  def _doBackprop(self,env,gradAccum,pad):
    env.delta[self.src] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src2])
    env.delta[self.src2] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src])
  def copy(self):
    return ComponentwiseVecMulOp(self.dst,self.src,self.src2)

class WeightedVec(Op):
  """Implements dst = vec * weighter.sum(), where dst and vec are row
  vectors.
  """
  def __init__(self,dst,weighter,vec):
    super(WeightedVec,self).__init__(dst)
    self.weighter = weighter
    self.vec = vec
    #self.src = "[%s,%s]" % (weighter,vec)  #TODO: remove?
  def __repr__(self):
    return "WeightedVec(%s,%s.sum(),%s)" % (self.dst,self.weighter,self.vec)
  def _ppLHS(self):
    return "%s * %s.sum()" % (self.vec,self.weighter)
  def _doEval(self,env,pad):
    env[self.dst] = mutil.broadcastAndWeightByRowSum(env[self.vec],env[self.weighter])
  def _doBackprop(self,env,gradAccum,pad):
    # This is written as a single operation
    #  dst = vec * weighter.sum()
    # but we will break into two steps conceptually
    #   1. weighterSum = weighter.sum()
    #   2. dst = vec * weighterSum
    # and then backprop through step 2, then step 1
    # step 2a: bp from delta[dst] to delta[vec]
    #   delta[vec] = delta[dst]*weighterSum
    env.delta[self.vec] = mutil.broadcastAndWeightByRowSum(env.delta[self.dst],env[self.weighter]) 
    # step 2b: bp from delta[dst] to delta[weighterSum]
    #   would be: delta[weighterSum] = (delta[dst].multiply(vec)).sum
    # followed by
    # step 1: bp from delta[weighterSum] to weighter
    #   delta[weighter] = delta[weighterSum]*weighter
    # but we can combine 2b and 1 as follows (optimized):
    tmp = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.vec])
    env.delta[self.weighter] = mutil.broadcastAndWeightByRowSum(env[self.weighter], tmp)
  def copy(self):
    return WeightedVec(self.dst,self.weighter,self.vec)
