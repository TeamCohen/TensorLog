# (C) William W. Cohen and Carnegie Mellon University, 2016

import numpy
import logging

import mutil
import tlerr

# if true print ops as they are executed
TRACE = False
# if true print outputs of ops - only use this for tiny test cases
LONG_TRACE = False

OPTIMIZE_COMPONENT_MULTIPLY = False

MAXDEPTH=8

##############################################################################
#
# environment - holds either computed values, or subexpressions
#
##############################################################################

class Envir(object):
    """Holds a MatrixDB object and a group of variable bindings.
    Variables are used in message-passing.
    """
    def __init__(self,db):
        self.register = {}
        self.delta = {}
        self.db = db
    def bindList(self,vars,vals):
        """Bind each variable in a list to the corresponding value."""
        assert len(vars)==len(vals)
        for i in range(len(vars)):
            self[vars[i]] = vals[i]
    def __repr__(self):
        return 'Envir(%r)' % self.register
    #override env[var] to access the binding array
    def __getitem__(self,key):
        return self.register[key]
    def __setitem__(self,key,val):
        self.register[key] = val

##############################################################################
#
# operators
#
##############################################################################

class Op(object):
    """Sort of like a function but side-effects an environment.  More
    specifically, this is the tensorlog encoding for matrix-db
    'operations' which can be 'eval'ed or differentiated. Operations
    typically specify src and dst variable names and eval-ing them
    will side-effect an environment, by binding the dst to some
    function of the input (src) bindings.
    """
    
    def __init__(self,dst):
        self.dst = dst
        self.msgFrom = self.msgTo = None
        self.src = "n/a"
    def setMessage(self,msgFrom,msgTo):
        """For debugging/tracing, record the BP message associated with this
        operation."""
        self.msgFrom = msgFrom
        self.msgTo = msgTo
    #TODO docstrings
    #TODO make this like what was done in funs.py, with _doEval and _doBackprop
    def eval(self,env):
        assert False,'abstract method called'
    def backprop(self,env,gradAccum):
        #these should all call eval first
        assert False,'abstract method called'
    def traceEvalCompletion(self,env):
        # call at end of eval
        if TRACE: 
            print 'op eval',self,
            if LONG_TRACE: print 'stores',env.db.matrixAsSymbolDict(env[self.dst])
            else: print
    def traceBackPropCompletion(self,env):
        # call at end of backprop
        if TRACE: 
            print 'op bp',self,'delta[',self.dst,']',
            if LONG_TRACE: print env.db.matrixAsSymbolDict(env.delta[self.dst])
            else: print
    def showDeltaShape(self,env,key):
        print 'shape of env.delta[%s]' % key,env.delta[key].get_shape()
    def showShape(self,env,key):
        print 'shape of env[%s]' % key,env[key].get_shape()
    def pprint(self,depth=-1):
        buf = '%s = %s' % (self.dst,self._ppLHS())
        if self.msgFrom and self.msgTo:
            buf = "%-45s // %s -> %s" % (buf,self.msgFrom,self.msgTo)
            return ["%s%-94s [%s]" % (('| '*depth),buf,self.__class__.__name__)]
        else:
            return ["%s%s" % (('| '*depth),buf)]
    def _ppLHS(self):
        #override
        return repr(self)

class DefinedPredOp(Op):
    """Op that calls a defined predicate."""
    def __init__(self,tensorlogProg,dst,src,mode,depth):
        super(DefinedPredOp,self).__init__(dst)
        self.tensorlogProg = tensorlogProg
        self.src = src
        self.funMode = mode
        self.depth = depth
    def __repr__(self):
        return "DefinedPredOp(%r,%r,%s,%d)" % (self.dst,self.src,str(self.funMode),self.depth)
    def _ppLHS(self):
        return "f_[%s,%d](%s)" % (str(self.funMode),self.depth,self.src)
    def eval(self,env):
        subfun = self.tensorlogProg.function[(self.funMode,self.depth)]
        vals = [env[self.src]]
        outputs = subfun.eval(self.tensorlogProg.db, vals)
        env[self.dst] = outputs
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        subfun = self.tensorlogProg.function[(self.funMode,self.depth)]
        foo = env.delta[self.dst]
        newDelta = subfun.backprop(env.delta[self.dst],gradAccum)
        if newDelta == None: raise tlerr.InvalidBackpropState("None delta received from %s\ndst %s, src %s\ndelta was: %s" % (subfun.__class__.__name__,self.dst,self.src,env.delta))
        env.delta[self.src] = newDelta
        logging.debug("%s(%s,%s) delta[%s] set to %s" % (self.__class__.__name__,self.dst,self.src,self.src,str(newDelta)))
        self.traceBackPropCompletion(env)
    def pprint(self,depth=-1):
        top = super(DefinedPredOp,self).pprint(depth)
        #return top
        #if self.depth>MAXDEPTH: return top
        if depth>MAXDEPTH: return top + ["%s..." % ('| '*(depth+1))]
        return top + self.tensorlogProg.function[(self.funMode,self.depth)].pprint(depth=depth+1)
            

class AssignPreimageToVar(Op):
    """Mat is something like p(X,Y) where Y is not used 'downstream' or
    p(X,c) where c is a constant.  Assign a row vector which encodes
    the preimage of the function defined by X to the environment
    variable 'dst'. """
    def __init__(self,dst,matMode):
        super(AssignPreimageToVar,self).__init__(dst)
        self.matMode = matMode
    def __repr__(self):
        return "AssignPreimageToVar(%s,%s)" % (self.dst,self.matMode)
    def _ppLHS(self):
        return "M_[%s]" % str(self.matMode)
    def eval(self,env):
        env[self.dst] = env.db.matrixPreimage(self.matMode)
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        #TODO implement preimages
        assert False,'backprop with preimages not implemented'

class AssignVectorToVar(Op):
    """Mat is a unary predicate like p(X). Assign a row vector which
    encodes p to the variable 'dst'. """
    def __init__(self,dst,matMode):
        super(AssignVectorToVar,self).__init__(dst)
        self.matMode = matMode
    def __repr__(self):
        return "AssignVectorToVar(%s,%s)" % (self.dst,self.matMode)
    def _ppLHS(self):
        return "V_[%s]" % str(self.matMode)
    def eval(self,env):
        env[self.dst] = env.db.vector(self.matMode)
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        if env.db.isParameter(self.matMode):        
            update = env.delta[self.dst]
            key = (self.matMode.functor,self.matMode.arity)
            gradAccum.accum(key,update)
        self.traceBackPropCompletion(env)

class AssignZeroToVar(Op):
    """Set the dst variable to an all-zeros row."""
    def __init__(self,dst):
        super(AssignZeroToVar,self).__init__(dst)
    def __repr__(self):
        return "ClearVar(%r)" % (self.dst)
    def _ppLHS(self):
        return "0"
    def eval(self,env):
        env[self.dst] = env.db.zeros()
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        self.traceBackPropCompletion(env)

class AssignOnehotToVar(Op):
    """ Assign a one-hot row encoding of a constant to the dst variable.
    """
    def __init__(self,dst,mode):
        super(AssignOnehotToVar,self).__init__(dst)
        self.onehotConst = mode.arg(1)
    def __repr__(self):
        return "AssignOnehotToVar(%s,%s)" % (self.dst,self.onehotConst)
    def _ppLHS(self):
        return 'U_[%s]' % self.onehotConst
    def eval(self,env):
        env[self.dst] = env.db.onehot(self.onehotConst)
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        self.traceBackPropCompletion(env)
    

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
    def eval(self,env):
        env[self.dst] = env[self.src] * env.db.matrix(self.matMode,self.transpose)
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        # dst = f(src,mat)
        try:
            env.delta[self.src] = env.delta[self.dst] * env.db.matrix(self.matMode,(not self.transpose))
            logging.debug("%s delta[%s] set to %s" % (self.__class__.__name__,self.src,mutil.summary(env.delta[self.src])))
        except Exception as e:
            def showmat(msg,m): print msg,type(m),m.get_shape(),m.nnz
            showmat(self.dst+' delta',env.delta[self.dst])
            showmat(str(self.matMode) + ' transpose=%r' % (not self.transpose ), env.db.matrix(self.matMode,(not self.transpose)))
            print e

        if env.db.isParameter(self.matMode):
            update = env[self.src].transpose() * (env.delta[self.dst])
            # The transpose flag is set in BP when sending a message
            # 'backward' from a goal output to variable, an indicates
            # if the operation needs to transpose the matrix.  Since
            # the db stores predicates p(a,b) internally as a matrix
            # where a is a row and b is a column, when the matMode is
            # p(o,i) then another internal transposition happens, by
            # the database.  We need to transpose the update when
            # exactly one of these transpositions happen, not two or
            # zero
            transposeUpdate = env.db.transposeNeeded(self.matMode,self.transpose)
            if transposeUpdate: update=update.transpose()
            # finally save the update
            key = (self.matMode.functor,self.matMode.arity)
            gradAccum.accum(key,update)
        self.traceBackPropCompletion(env)

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
    def eval(self,env):
        if OPTIMIZE_COMPONENT_MULTIPLY:
            env[self.dst] = mutil.broadcastAndComponentwiseMultiply(env[self.src],env[self.src2])
        else:
            m1,m2 = mutil.broadcastBinding(env,self.src,self.src2)
            env[self.dst] = m1.multiply(m2)
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        if OPTIMIZE_COMPONENT_MULTIPLY:
            env.delta[self.src] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src2])
            env.delta[self.src2] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src])
        else:
            m1,m2 = mutil.broadcastBinding(env,self.src,self.src2)            
            env.delta[self.src] = env.delta[self.dst].multiply(m2)
            logging.debug("%s delta[%s] set to %s" % (self.__class__.__name__,self.src,mutil.summary(env.delta[self.src])))
            env.delta[self.src2] = env.delta[self.dst].multiply(m1)
            logging.debug("%s delta[%s] set to %s" % (self.__class__.__name__,self.src2,mutil.summary(env.delta[self.src2])))
        self.traceBackPropCompletion(env)

class WeightedVec(Op):
    """Implements dst = vec * weighter.sum(), where dst and vec are row
    vectors.
    """
    def __init__(self,dst,weighter,vec):
        super(WeightedVec,self).__init__(dst)
        self.weighter = weighter
        self.vec = vec
        self.src = "[%s,%s]" % (weighter,vec)
    def __repr__(self):
        return "WeightedVec(%s,%s.sum(),%s)" % (self.dst,self.weighter,self.vec)
    def _ppLHS(self):
        return "%s * %s.sum()" % (self.vec,self.weighter)
    def eval(self,env):
        #optimized
        #m1,m2 = mutil.broadcastBinding(env, self.vec, self.weighter)
        #env[self.dst] = mutil.weightByRowSum(m1,m2)
        env[self.dst] = mutil.broadcastAndWeightByRowSum(env[self.vec],env[self.weighter])
        self.traceEvalCompletion(env)
    def backprop(self,env,gradAccum):
        # This is written as a single operation
        #    dst = vec * weighter.sum()
        # but we will break into two steps conceptually
        #   1. weighterSum = weighter.sum()
        #   2. dst = vec * weighterSum
        # and then backprop through step 2, then step 1
        # step 2a: bp from delta[dst] to delta[vec]
        # old slow version was:
        #   mVec,mWeighter = mutil.broadcastBinding(env, self.vec, self.weighter)
        #   env.delta[self.vec] = mutil.weightByRowSum(env.delta[self.dst],mWeighter)
        # new optimized version of step 2a
        env.delta[self.vec] = mutil.broadcastAndWeightByRowSum(env.delta[self.dst],env[self.weighter]) 
        logging.debug("%s delta[%s] set to %s" % (self.__class__.__name__,self.vec,mutil.summary(env.delta[self.vec])))
        # step 2b: bp from delta[dst] to delta[weighterSum]
        #   would be: delta[weighterSum] = (delta[dst].multiply(vec)).sum
        # followed by 
        # step 1: bp from delta[weighterSum] to weighter
        #   delta[weighter] = delta[weighterSum]*weighter
        # but we can combine 2b and 1 as follows (optimized):
        env.delta[self.weighter] = mutil.broadcastAndWeightByRowSum(env[self.weighter], env.delta[self.dst].multiply(env[self.vec]))
        logging.debug("%s delta[%s] set to %s" % (self.__class__.__name__,self.weighter,mutil.summary(env.delta[self.vec])))
        self.traceBackPropCompletion(env)
