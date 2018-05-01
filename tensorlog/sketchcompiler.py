from tensorlog import bpcompiler as bc
from tensorlog import ops,mutil,funs
import scipy.sparse

class SketchCompiler(bc.BPCompiler):
  """Compiles a logical rule + a mode into a sequence of ops.py operations."""
  def __init__(self,lhsMode,tensorlogProg,depth,rule,sketch):
    self.sk = sketch
    super(SketchCompiler,self).__init__(lhsMode, tensorlogProg, depth, rule)
  def _initOperatorImplementations(self):
    bc.BPCompiler._initOperatorImplementations(self)
    self.opImpl['AssignPreimageToVar'] = AssignPreimageSketchToVar.factory(self.sk)
    self.opImpl['AssignVectorToVar']   = AssignVectorSketchToVar.factory(self.sk)
    self.opImpl['AssignOnehotToVar']   = AssignOnehotSketchToVar.factory(self.sk)
    self.opImpl['VecMatMulOp']         = FollowOp.factory(self.sk)
    self.opImpl['WeightedVec']         = SketchWeightedVec.factory(self.sk)

class AssignPreimageSketchToVar(ops.AssignPreimageToVar):
  def __init__(self,dst,matMode,sketch,dstType=None):
    super(AssignPreimageSketchToVar,self).__init__(dst, matMode, dstType)
    self.sk = sketch
  def _doEval(self,env,pad):
    env[self.dst] = self.sk.sketch(env.db.matrixPreimage(self.matMode))
  def _doBackprop(self,env,gradAccum,pad):
    #TODO implement preimages
    assert False,'backprop with preimages not implemented'
  def copy(self):
    return AssignPreimageSketchToVar(self.dst,self.matMode,self.sk)
  @staticmethod
  def factory(sk):
    def wrapper(dst,matMode,dstType=None):
      return AssignPreimageSketchToVar(dst,matMode,sk,dstType)
    return wrapper

class AssignVectorSketchToVar(ops.AssignVectorToVar):
  """Mat is a unary predicate like p(X). Assign a row vector which
  encodes p to the variable 'dst'. """
  def __init__(self,dst,matMode,sketch,dstType=None):
    super(AssignVectorSketchToVar,self).__init__(dst, matMode, dstType)
    self.sk = sketch
  def _doEval(self,env,pad):
    try:
      env[self.dst] = self.sk.sketch(env.db.vector(self.matMode))
    except:
      print type(self),self.matMode
      raise
  def _doBackprop(self,env,gradAccum,pad):
    if env.db.isParameter(self.matMode):
      update = env.delta[self.dst]
      key = (self.matMode.functor,self.matMode.arity)
      gradAccum.accum(key,update)
  def copy(self):
    return AssignVectorSketchToVar(self.dst,self.matMode,self.sk)
  @staticmethod
  def factory(sk):
    def wrapper(dst, matMode, dstType=None):
      return AssignVectorSketchToVar(dst, matMode,sk,dstType)
    return wrapper
  
class AssignOnehotSketchToVar(ops.AssignOnehotToVar):
  """Assign a one-hot row encoding of a constant to the dst variable.
  Mode is either assign(var,const) or assign(var,const,type)
  """
  def __init__(self,dst,mode,sketch):
    super(AssignOnehotSketchToVar,self).__init__(dst,mode)
    self.sk = sketch
  def _doEval(self,env,pad):
    env[self.dst] = self.sk.sketch(env.db.onehot(self.onehotConst,self.dstType))
  def _doBackprop(self,env,gradAccum,pad):
    pass
  def copy(self):
    return AssignOnehotSketchToVar(self.dst,self.mode,self.sk)
  @staticmethod
  def factory(sk):
    def wrapper(dst,mode):
      return AssignOnehotSketchToVar(dst,mode,sk)
    return wrapper

class FollowOp(ops.VecMatMulOp):
  """Op of the form "dst = src*mat or dst=src*mat.tranpose()"
  """
  def __init__(self,dst,src,matMode,sketch,transpose=False):
    super(FollowOp,self).__init__(dst,src,matMode)
    self.sk = sketch
  def _doEval(self,env,pad):
    env[self.dst] = self.sk.follow(self.matMode, env[self.src], self.transpose)
  def _doBackprop(self,env,gradAccum,pad):
    # dst = f(src,mat)
    env.delta[self.src] = self.sk.follow(self.matMode, env.delta[self.dst], (not self.transpose))
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
    return FollowOp(self.dst,self.src,self.matMode,self.sk, self.transpose)
  @staticmethod
  def factory(sk):
    def wrapper(dst,src,matMode,transpose=False):
      return FollowOp(dst,src,matMode,sk,transpose)
    return wrapper


class SketchWeightedVec(ops.WeightedVec):
  """Implements dst = vec * weighter.sum(), where dst and vec are row
  vectors.
  """
  def __init__(self,dst,weighter,vec,sketch):
    super(SketchWeightedVec,self).__init__(dst,weighter,vec)
    self.sk = sketch
    #self.src = "[%s,%s]" % (weighter,vec)  #TODO: remove?
  def _doEval(self,env,pad):
    env[self.dst] = mutil.broadcastAndWeightByRowSum(env[self.vec],env[self.weighter]/self.sk.t)
  def _doBackprop(self,env,gradAccum,pad):
    # This is written as a single operation
    #  dst = vec * weighter.sum()
    # but we will break into two steps conceptually
    #   1. weighterSum = weighter.sum()
    #   2. dst = vec * weighterSum
    # and then backprop through step 2, then step 1
    # step 2a: bp from delta[dst] to delta[vec]
    #   delta[vec] = delta[dst]*weighterSum
    env.delta[self.vec] = mutil.broadcastAndWeightByRowSum(env.delta[self.dst],env[self.weighter]/self.sk.t) 
    # step 2b: bp from delta[dst] to delta[weighterSum]
    #   would be: delta[weighterSum] = (delta[dst].multiply(vec)).sum
    # followed by
    # step 1: bp from delta[weighterSum] to weighter
    #   delta[weighter] = delta[weighterSum]*weighter
    # but we can combine 2b and 1 as follows (optimized):
    tmp = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.vec])
    env.delta[self.weighter] = mutil.broadcastAndWeightByRowSum(env[self.weighter], tmp)
  def copy(self):
    return SketchWeightedVec(self.dst,self.weighter,self.vec)
  @staticmethod
  def factory(sk):
    def wrapper(dst,weighter,vec):
      return SketchWeightedVec(dst,weighter,vec,sk)
    return wrapper

import math
import numpy as NP
def softmax(db,mat,sk):
    """ Compute the softmax of each row of a matrix.
    """
    nullEpsilon = -10  # scores for null entity will be exp(nullMatrix)
    result = sk.sketch(db.nullMatrix(mutil.numRows(mat))*nullEpsilon) 
    #print db.matrixAsSymbolDict(sk.unsketch(result))
    #print mutil.pprintSummary(result)
    #print mutil.pprintSummary(mat)
    result = result + mat
    #print mutil.pprintSummary(result)
    #print db.matrixAsSymbolDict(sk.unsketch(mat))
    #print db.matrixAsSymbolDict(sk.unsketch(result))
    #denseResult,undensifier = mutil.densify(result)
    #print db.matrixAsSymbolDict(sk.unsketch(denseResult))
    denseResult = undensifier = None
    if not (denseResult is None):
        #print "using denseSoftmax"
        result = mutil.undensify(mutil.denseSoftmax(denseResult), undensifier)
        return result * sk.t
    else:
        def softMaxAlteration(data,lo,hi,unused):
            rowMax = max(data[lo:hi])
            assert not math.isnan(rowMax),"softMaxAlteration: NaN rowMax"
            data[lo:hi] = NP.exp(data[lo:hi] - rowMax)
            rowNorm = sum(data[lo:hi]) / sk.t
            assert not math.isnan(rowNorm),"softMaxAlteration: NaN rowNorm"
            data[lo:hi] /= rowNorm
            #replace the zeros in data, which are underflow, with something small
            minValue = math.exp(nullEpsilon)
            segment = data[lo:hi]
            segment[segment==0] = minValue
            data[lo:hi] = segment
        mutil.alterMatrixRows(result,softMaxAlteration)
        return result

class SketchSoftmaxFunction(funs.SoftmaxFunction):
    """A function which computes row-wise softmax of an inner function."""

    def __init__(self,fun,sketch):
        super(SketchSoftmaxFunction,self).__init__(fun)
        self.sketcher = sketch
    def _doEval(self,db,values,pad):
        unnorm = self.fun.eval(db,values,pad)
        #print db.matrixAsSymbolDict(self.sketcher.unsketch(unnorm))
        result = softmax(db,unnorm,self.sketcher)
        return result
    def copy(self):
        return SketchSoftmaxFunction(self.fun.copy())
