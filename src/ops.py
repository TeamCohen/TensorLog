# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import numpy
import logging

import bcast

# if true print ops as they are executed
# TODO: abstract the trace routine?

def trace(): return logging.getLogger().isEnabledFor(logging.DEBUG)

#TODO shorter name?
class Partial(object):
    """Encapsulate a variable name for the partial derivative of f wrt x"""
    def __init__(self,f,x):
        self.f = f
        self.x = x
    def __str__(self):
        return "d%s/d%s" % (self.f,self.x)
    def __repr__(self):
        return "Partial(%r,%r)" % (self.f,self.x)
    def __hash__(self):
        return hash((self.f,self.x))
    def __eq__(self,other):
        return self.f==other.f and self.x==other.x

def paramMatchMode(w,mode):
    (p,k) = w
    return (p==mode.functor and k==mode.arity)
    

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
    def eval(self,env):
        assert False,'abstract method called'
    def evalGrad(self,env):
        #these should all call eval first
        assert False,'abstract method called'

class DefinedPredOp(Op):
    """Op that calls a defined predicate."""
    def __init__(self,tensorlogProg,dst,src,mode,depth):
        self.tensorlogProg = tensorlogProg
        self.dst = dst
        self.src = src
        self.mode = mode
        self.depth = depth
    def __str__(self):
        return "DefinedPredOp<%s = %s(%s,%d)>" % (self.dst,self.mode,self.src,self.depth)
    def __repr__(self):
        return "DefinedPredOp(%r,%r,%s,%d)" % (self.dst,self.src,str(self.mode),self.depth)
    def eval(self,env):
        subfun = self.tensorlogProg.function[(self.mode,self.depth)]
        vals = [env[self.src]]
        outputs = subfun.eval(self.tensorlogProg.db, vals)
        env[self.dst] = outputs[0]
    def evalGrad(self,env):
        subfun = self.tensorlogProg.function[(self.mode,self.depth)]
        vals = [env[self.src]]
        gradDict = subfun.evalGrad(self.tensorlogProg.db, vals)
        for var,val in gradDict.items():
            env[var] = val

class AssignPreimageToVar(Op):
    """Mat is a like p(X,Y) where Y is not used 'downstream' or p(X,c)
    where c is a constant.  Assign a row vector which encodes the
    preimage of the function defined by X to the environment variable
    'dst'. """
    def __init__(self,dst,matMode):
        self.dst = dst
        self.matMode = matMode
    def __str__(self):
        return "Assign(%s = preimage(%s))" % (self.dst,self.matMode)
    def __repr__(self):
        return "AssignPreimageToVar(%s,%s)" % (self.dst,self.matMode)
    def eval(self,env):
        if trace(): print 'op:',self
        env[self.dst] = env.db.matrixPreimage(self.matMode)
        if trace(): print self.dst,'=>',env.db.matrixAsSymbolDict(env[self.dst])
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            if trace(): print 'evalGrad',self.dst,'/',w,'dict',env.keys()
            if paramMatchMode(w,self.matMode):
                env[Partial(self.dst,w)] = env.db.ones()
            else:
                env[Partial(self.dst,w)] = env.db.zeros()

class AssignZeroToVar(Op):
    """Set the dst variable to an all-zeros row."""
    def __init__(self,dst):
        self.dst = dst
    def __str__(self):
        return "ClearVar(%s)" % (self.dst)
    def __repr__(self):
        return "ClearVar(%r)" % (self.dst)
    def eval(self,env):
        if trace(): print 'op:',self
        env[self.dst] = env.db.zeros()
        if trace(): print self.dst,'=>',env.db.matrixAsSymbolDict(env[self.dst])
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            env[Partial(self.dst,w)] = env.db.ones()

class AssignOnehotToVar(Op):
    """Mat is a like p(X,Y) where Y is not used 'downstream' or p(X,c)
    where c is a constant.  Assign a row vector which encodes the
    preimage of the function defined by X to the environment variable
    'dst'. """
    def __init__(self,dst,mode):
        self.dst = dst
        self.onehotConst = mode.arg(1)
    def __str__(self):
        return "AssignOnehotToVar(%s = preimage(%s))" % (self.dst,self.onehotConst)
    def __repr__(self):
        return "AssignOnehotToVar(%s,%s)" % (self.dst,self.onehotConst)
    def eval(self,env):
        if trace(): print 'op:',self
        env[self.dst] = env.db.onehot(self.onehotConst)
        if trace(): print self.dst,'=>',env.db.matrixAsSymbolDict(env[self.dst])
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            env[Partial(self.dst,w)] = env.db.ones()

class VecMatMulOp(Op):
    """Op of the form "dst = src*mat or dst=src*mat.tranpose()"
    """
    def __init__(self,dst,src,matmode,transpose=False):
        self.dst = dst
        self.src = src
        self.matmode = matmode
        self.transpose = transpose
    def __str__(self):
        transFlag = ".transpose()" if self.transpose else ""
        return "VecMatMulOp<%s = %s * %s%s>" % (self.dst,self.src,self.matmode,transFlag)
    def __repr__(self):
        return "VecMatMulOp(%r,%r,%s,%r)" % (self.dst,self.src,self.matmode,self.transpose)
    def eval(self,env):
        if trace(): print 'op:',self
        env[self.dst] = env[self.src] * env.db.matrix(self.matmode,self.transpose)
        if trace(): print self.dst,'=>',env.db.matrixAsSymbolDict(env[self.dst])
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            if trace(): print 'evalGrad',self.dst,'/',w,'dict',env.keys()
            if paramMatchMode(w,self.matmode):
                # df/dp r*M = (df/dp r) * M + r (df/dp M)
                #           = (df/dp r) * M + r I            if p==M
                #           = (df/dp r) * M + r I            else
                env[Partial(self.dst,w)] = \
                    env[Partial(self.src,w)] * env.db.matrix(self.matmode,self.transpose)  + env[self.src] 
            else:
                env[Partial(self.dst,w)] = \
                    env[Partial(self.src,w)] * env.db.matrix(self.matmode,self.transpose)
                
#
# the ones that are tricky with minibatch inputs
#

class ComponentwiseVecMulOp(Op):
    """ Computes dst = src*Diag(src2), i.e., the component-wise product of
    two row vectors.  
    """
    def __init__(self,dst,src,src2):
        self.dst = dst
        self.src = src
        self.src2 = src2
    def __str__(self):
        return "ComponentwiseVecMulOp<%s = %s * %s>" % (self.dst,self.src,self.src2)
    def __repr__(self):
        return "ComponentwiseVecMulOp(%r,%r,%s)" % (self.dst,self.src,self.src2)
    def eval(self,env):
        if trace(): 
            print 'op:',self
        m1,m2 = bcast.broadcastBinding(env, self.src, self.src2)
        env[self.dst] = m1.multiply(m2)
        if trace(): print self.dst,'=>',env.db.matrixAsSymbolDict(env[self.dst])
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            m1a,m2a = bcast.broadcastBinding(env, Partial(self.src,w), self.src2)
            m1b,m2b = bcast.broadcastBinding(env, self.src, Partial(self.src2,w))
            m1a,m2a,m1b,m2b = bcast.broadcast4(m1a,m2a,m1b,m2b)
            env[Partial(self.dst,w)] = m1a.multiply(m2a) + m1b.multiply(m2b)

#TODO move some of this logic into bcast so I can optimize later

class WeightedVec(Op):
    """Implements dst = vec * weighter.sum(), where dst and vec are row
    vectors.
    """
    def __init__(self,dst,weighter,vec):
        self.dst = dst
        self.weighter = weighter
        self.vec = vec
    def __str__(self):
        return "WeightedVec<%s = %s * %s>" % (self.dst,self.weighter,self.vec)
    def __repr__(self):
        return "WeightedVec<%s,%s,%s>" % (self.dst,self.weighter,self.vec)
    def eval(self,env):
        m1,m2 = bcast.broadcastBinding(env, self.vec, self.weighter)
        r = bcast.numRows(m1)  #also m2
        if r==1:
            env[self.dst] =  m1 * m2.sum()
        else:
            env[self.dst] =  \
                SS.vstack([m2.getrow(i).sum() * m1.getrow(i) for i in range(r)], dtype='float64')
    def evalGrad(self,env):
        self.eval(env)
        for w in env.db.params:
            m1a,m2a = bcast.broadcastBinding(env, Partial(self.vec,w), self.weighter)
            m1b,m2b = bcast.broadcastBinding(env,self.vec,Partial(self.weighter,w))
            m1a,m2a,m1b,m2b = bcast.broadcast4(m1a,m2a,m1b,m2b)
            r = bcast.numRows(m1a)
            if r==1:
                env[Partial(self.dst,w)] =  m1a*m2a.sum() + m1b*m2b.sum()
            else:
                env[Partial(self.dst,w)] = \
                    SS.vstack([m2a.getrow(i).sum() * m1a.getrow(i) for i in range(r)], dtype='float64') \
                    + SS.vstack([m2b.getrow(i).sum() * m1b.getrow(i) for i in range(r)], dtype='float64')
