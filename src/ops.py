# (C) William W. Cohen and Carnegie Mellon University, 2016

import numpy
import logging
import scipy.sparse

import mutil
import config

conf = config.Config()
conf.trace = False;                     conf.help.trace =                       "Print debug info during op execution"
conf.long_trace = False;                conf.help.long_trace =                  "Print output of functions after op - only for small tasks"
conf.optimize_weighted_vec=True;        conf.help.optimize_weighted_vec =       "Use optimized version of WeightedVec op"
conf.optimize_component_multiply=True;  conf.help.optimize_component_multiply = "Use optimized version of ComponentwiseVecMulOp"

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

def isBuiltinIOOp(mode):
    return mode.functor=='printf'

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
    def setMessage(self,msgFrom,msgTo):
        """For debugging/tracing, record the BP message associated with this
        operation."""
        self.msgFrom = msgFrom
        self.msgTo = msgTo
    #TODO docstrings
    def eval(self,env):
        if conf.trace:
            print 'eval',self,
        self._doEval(env)
        if conf.trace:
            if conf.long_trace: print 'stores',env.db.matrixAsSymbolDict(env[self.dst])
            else: print
    def backprop(self,env,gradAccum):
        #these should all call eval first
        if conf.trace:
            print 'call bp',self,'delta[',self.dst,'] shape',env.delta[self.dst].get_shape(),
            if conf.long_trace: print env.db.matrixAsSymbolDict(env.delta[self.dst])
            else: print
        self._doBackprop(env,gradAccum)
        if conf.trace: 
            print 'end bp',self
    def showDeltaShape(self,env,key):
        print 'shape of env.delta[%s]' % key,env.delta[key].get_shape()
    def showShape(self,env,key):
        print 'shape of env[%s]' % key,env[key].get_shape()
    def pprint(self):
        buf = '%s = %s' % (self.dst,self._ppLHS())
        if self.msgFrom and self.msgTo:
            return '%-45s // %s -> %s' % (buf,self.msgFrom,self.msgTo)
        else:
            return buf
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
    def _doEval(self,env):
        subfun = self.tensorlogProg.function[(self.funMode,self.depth)]
        vals = [env[self.src]]
        outputs = subfun.eval(self.tensorlogProg.db, vals)
        env[self.dst] = outputs
    def _doBackprop(self,env,gradAccum):
        subfun = self.tensorlogProg.function[(self.funMode,self.depth)]
        newDelta = subfun.backprop(env.delta[self.dst],gradAccum)
        env.delta[self.src] = newDelta

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
    def _doEval(self,env):
        env[self.dst] = env.db.matrixPreimage(self.matMode)
    def _doBackprop(self,env,gradAccum):
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
    def _doEval(self,env):
        env[self.dst] = env.db.vector(self.matMode)
    def _doBackprop(self,env,gradAccum):
        if env.db.isParameter(self.matMode):        
            update = env.delta[self.dst]
            key = (self.matMode.functor,self.matMode.arity)
            gradAccum.accum(key,update)

class AssignZeroToVar(Op):
    """Set the dst variable to an all-zeros row."""
    def __init__(self,dst):
        super(AssignZeroToVar,self).__init__(dst)
    def __repr__(self):
        return "ClearVar(%r)" % (self.dst)
    def _ppLHS(self):
        return "0"
    def _doEval(self,env):
        #TODO - zeros(n)? or is this used now?
        env[self.dst] = env.db.zeros()
    def _doBackprop(self,env,gradAccum):
        #TODO check
        pass

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
    def _doEval(self,env):
        env[self.dst] = env.db.onehot(self.onehotConst)
    def _doBackprop(self,env,gradAccum):
        pass

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
    def _doEval(self,env):
        env[self.dst] = env[self.src] * env.db.matrix(self.matMode,self.transpose)
    def _doBackprop(self,env,gradAccum):
        # dst = f(src,mat)
        env.delta[self.src] = env.delta[self.dst] * env.db.matrix(self.matMode,(not self.transpose))
        mutil.checkCSR(env.delta[self.src],'delta[%s]' % self.src)
        if env.db.isParameter(self.matMode):
            update = env[self.src].transpose() * (env.delta[self.dst])
            update = scipy.sparse.csr_matrix(update)
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
            if transposeUpdate: 
                update = update.transpose()
                update = scipy.sparse.csr_matrix(update)
            # finally save the update
            key = (self.matMode.functor,self.matMode.arity)
            mutil.checkCSR(update,'update for %s mode %s transpose %s' % (str(key),str(self.matMode),transposeUpdate))
            gradAccum.accum(key,update)

class BuiltInIOOp(Op):
    """Built-in special op, like printf(src,dst), with one input and one
    output variable.
    """
    def __init__(self,dst,src,matMode):
        super(BuiltInIOOp,self).__init__(dst)
        self.src = src
        self.matMode = matMode
    def __repr__(self):
        return "BuiltInIOOp(%r,%r,%s)" % (self.dst,self.src,self.matMode)
    def _ppLHS(self):
        return "buitin_%s(%s)" % (self.matMode.functor,self.src)
    def _doEval(self,env):
        assert self.matMode.functor=='printf'
        d = env.db.matrixAsSymbolDict(env[self.src])
        print '= %s->%s' % (self.msgFrom,self.msgTo),
        if len(d.keys())==1:
            print d[0]
        else:
            print
            for k in d:
                print '= row',k,'=>',d[k]
        env[self.dst] = env[self.src]
    def _doBackprop(self,env,gradAccum):
        env.delta[self.src] = env.delta[self.dst]


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
    def _doEval(self,env):
        if conf.optimize_component_multiply:
            env[self.dst] = mutil.broadcastAndComponentwiseMultiply(env[self.src],env[self.src2])
        else:
            m1,m2 = mutil.broadcastBinding(env,self.src,self.src2)
            env[self.dst] = m1.multiply(m2)
    def _doBackprop(self,env,gradAccum):
        if conf.optimize_component_multiply:
            env.delta[self.src] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src2])
            env.delta[self.src2] = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.src])
        else:
            #m1,m2 = mutil.broadcastBinding(env,self.src,self.src2)            
            #print 'shapes 1/2/delta[',self.dst,']',m1.get_shape(),m2.get_shape(),env.delta[self.dst].get_shape()
            d1,m1 = mutil.broadcast2(env.delta[self.dst],env[self.src])
            d2,m2 = mutil.broadcast2(env.delta[self.dst],env[self.src2])
            assert d1.get_shape()==d2.get_shape()
            env.delta[self.src] = d2.multiply(m2)
            env.delta[self.src2] = d1.multiply(m1)
#            env.delta[self.src] = env.delta[self.dst].multiply(m2)
#            env.delta[self.src2] = env.delta[self.dst].multiply(m1)

class WeightedVec(Op):
    """Implements dst = vec * weighter.sum(), where dst and vec are row
    vectors.
    """
    def __init__(self,dst,weighter,vec):
        super(WeightedVec,self).__init__(dst)
        self.weighter = weighter
        self.vec = vec
    def __repr__(self):
        return "WeightedVec(%s,%s.sum(),%s)" % (self.dst,self.weighter,self.vec)
    def _ppLHS(self):
        return "%s * %s.sum()" % (self.vec,self.weighter)
    def _doEval(self,env):
        if conf.optimize_weighted_vec:
            env[self.dst] = mutil.broadcastAndWeightByRowSum(env[self.vec],env[self.weighter])
        else:
            m1,m2 = mutil.broadcastBinding(env, self.vec, self.weighter)
            env[self.dst] = mutil.weightByRowSum(m1,m2)
    def _doBackprop(self,env,gradAccum):
        # This is written as a single operation
        #    dst = vec * weighter.sum()
        # but we will break into two steps conceptually
        #   1. weighterSum = weighter.sum()
        #   2. dst = vec * weighterSum
        # and then backprop through step 2, then step 1
        # step 2a: bp from delta[dst] to delta[vec]
        #   delta[vec] = delta[dst]*weighterSum
        if conf.optimize_weighted_vec:
            env.delta[self.vec] = mutil.broadcastAndWeightByRowSum(env.delta[self.dst],env[self.weighter]) 
        else:
            deltaDst,mWeighter = mutil.broadcast2(env.delta[self.dst], env[self.weighter])
            env.delta[self.vec] = mutil.weightByRowSum(deltaDst,mWeighter)
        # step 2b: bp from delta[dst] to delta[weighterSum]
        #   would be: delta[weighterSum] = (delta[dst].multiply(vec)).sum
        # followed by 
        # step 1: bp from delta[weighterSum] to weighter
        #   delta[weighter] = delta[weighterSum]*weighter
        # but we can combine 2b and 1 as follows (optimized):
        if conf.optimize_weighted_vec:
            if conf.optimize_component_multiply:
                tmp = mutil.broadcastAndComponentwiseMultiply(env.delta[self.dst],env[self.vec])
            else:
                m1,m2 = mutil.broadcast2(env.delta[self.dst],env[self.vec])
                tmp = m1.multiply(m2)
            env.delta[self.weighter] = \
                mutil.broadcastAndWeightByRowSum(env[self.weighter], tmp)
        else:
            #not clear if this still works
            m1,m2 = mutil.broadcast2(env.delta[self.dst],env[self.vec])
            tmp = m1.multiply(m2)
            mWeighter,mTmp = mutil.broadcast2(env[self.weighter],tmp)
            env.delta[self.weighter] = mutil.weightByRowSum(mWeighter,mTmp)

