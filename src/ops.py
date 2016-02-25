# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse
import numpy

TRACE=False

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

def isParamMode(db,mat):
    return (mat.functor,mat.arity) in db.params
    
##############################################################################
#
# environment - holds either computed values, or subexpressions
#
##############################################################################

class Envir(object):
    """Holds a MatrixDB object and a group of variable bindings.
    Variables are used in message-passing and are normally
    row matrices
    """
    def __init__(self,db):
        self.binding = {}
        self.db = db
    def bind(self,name,val):
        self.binding[name] = val
    def binding(self,name):
        return self.binding[name]
    def show():
        for v in self.binding:
            d = rowVarsAsSymbolDict(self,v)
            for r in d:
                print 'variable',v,'row',r,':'
                for s,w in d[r].items():
                    print '\t%s\t%g' % (s,w)

##############################################################################
#
# functions
#
##############################################################################

class Function(object):
    """The tensorlog representation of a function.  """
    def eval(self,db,values):
        """When called with a MatrixDB and a list of input values v1,...,xk,
        executes some function f(v1,..,vk) and return the output of f,
        which is a list of output values."""
        assert False, 'abstract method called.'
    def evalGrad(self,db,values):
        """Return a dictionary mapping Partial(f,w)=>the partial deriv of f
        wrt w for param w at the specified input values. Will also
        compute and return the value of the function
        """
        assert False, 'abstract method called.'
    def recurselyUse(self,pyfunction,db,values):
        """Implements an eval, if pyfunction is self.op.eval and the values
        are inputs. It binds input variables to values, in order,
        executes the inner op, and returns a dictionary mapping output
        variables to outputs.
        """
        assert False, 'abstract method called'


class OpFunction(Function):
    """A function defined by a single operator."""
    def __init__(self,inputs,outputs,op):
        self.inputs = inputs
        self.outputs = outputs
        self.op = op
    def __str__(self):
        return "Function(%r,%r,%r)" % (self.inputs,self.outputs,self.op)
    def __repr__(self):
        return "Function(%r,%r,%r)" % (self.inputs,self.outputs,self.op)
    def eval(self,db,values):
        env = self._envAfter(self.op.eval,db,values,Envir(db))
        return [env.binding[y] for y in self.outputs]
    def evalGrad(self,db,values):
        initEnv = Envir(db)
        for x in self.inputs:
            for p,k in db.params:
                initEnv.binding[Partial(x,(p,k))] = db.zeros()
        env = self._envAfter(self.op.evalGrad,db,values,initEnv)
        #collect the needed bindings in a dict
        gradDict = {}
        for y in self.outputs:
            gradDict[y] = env.binding[y]
            for w in env.db.params.keys():
                p_yw = Partial(y,w)
                gradDict[p_yw] = env.binding[p_yw]
        return gradDict
    def _envAfter(self,pyfun,db,values,env):
        for i,v in enumerate(values):
            env.binding[self.inputs[i]] = v
        pyfun(env)
        return env

class SumFunction(Function):
    """Sum of a bunch of functions."""
    def __init__(self,funs):
        self.funs = funs
    def __str__(self):
        return "(" + " + ".join(map(repr,self.funs)) + ")"
    def __repr__(self):
        return "SumFunction("+repr(self.funs)+")"
    def eval(self,db,values):
        baseValues = self.funs[0].eval(db,values)
        for f in self.funs[1:]:
            moreValues = f.eval(db,values)
            assert len(moreValues)==len(baseValues)
            for j in range(len(moreValues)):
                baseValues[j] = baseValues[j] + moreValues[j]
        return baseValues
    def evalGrad(self,db,values):
        baseDict = self.funs[0].evalGrad(db,values)
        constZeros = db.zeros()
        for f in self.funs[1:]:
            moreDict = f.evalGrad(db,values)
            for var,val in moreDict.items():
                baseDict[var] = baseDict.get(var,constZeros) + moreDict.get(var,constZeros)
        return baseDict

##############################################################################
#
# operators
#
##############################################################################

class Op(object):
    """Like a function but side-effects an environment.  More
    specifically, this is the tensorlog encoding for matrix-db
    'operations' which can be 'eval'ed or differentiated. Operations
    typically specify src and dst variable names and eval-ing them
    will side-effect an environment, by binding the src to some
    function of the dst's binding.
    """
    def eval(self,env):
        assert False,'abstract method called'
    def evalGrad(self,env):
        #these should all call eval first
        assert False,'abstract method called'

class SeqOp(object):
    """Sequence of other operations."""
    def __init__(self,ops):
        self.ops = ops
    def __str__(self):
        return "{" + "; ".join(map(str,self.ops)) + "}"
    def __repr__(self):
        return "SeqOp("+repr(self.ops)+")"
    def eval(self,env):
        for op in self.ops:
            op.eval(env)
    def evalGrad(self,env):
        self.eval(env)
        for op in self.ops:
            op.evalGrad(env)

# calls a function

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
        vals = [env.binding[self.src]]
        outputs = subfun.eval(self.tensorlogProg.db, vals)
        env.binding[self.dst] = outputs[0]
    def evalGrad(self,env):
        subfun = self.tensorlogProg.function[(self.mode,self.depth)]
        vals = [env.binding[self.src]]
        gradDict = subfun.evalGrad(self.tensorlogProg.db, vals)
        for var,val in gradDict.items():
            env.binding[var] = val

class AssignPreimageToVar(Op):
    """Mat is a like p(X,Y) where Y is not used 'downstream' or p(X,c)
    where c is a constant.  Assign a row vector which encodes the
    preimage of the function defined by X to the environment variable
    'dst'. """
    def __init__(self,dst,mat):
        self.dst = dst
        self.mat = mat
    def __str__(self):
        return "Assign(%s = preimage(%s))" % (self.dst,self.mat)
    def __repr__(self):
        return "AssignPreimageToVar(%s,%s)" % (self.dst,self.mat)
    def eval(self,env):
        if TRACE: print 'op:',self
        env.binding[self.dst] = env.db.matrixPreimage(self.mat)
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            if TRACE: print 'evalGrad',self.dst,'/',(p,k),'dict',env.binding.keys()
            if p==self.mat.functor and k==self.mat.arity:
                env.binding[Partial(self.dst,(p,k))] = env.db.ones()
            else:
                env.binding[Partial(self.dst,(p,k))] = env.db.zeros()

class AssignZeroToVar(Op):
    """Set the dst variable to an all-zeros row."""
    def __init__(self,dst):
        self.dst = dst
    def __str__(self):
        return "ClearVar(%s)" % (self.dst)
    def __repr__(self):
        return "ClearVar(%r)" % (self.dst)
    def eval(self,env):
        if TRACE: print 'op:',self
        env.binding[self.dst] = env.db.zeros()
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            env.binding[Partial(self.dst,(p,k))] = env.db.ones()

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
        if TRACE: print 'op:',self
        env.binding[self.dst] = env.db.onehot(self.onehotConst)
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            env.binding[Partial(self.dst,(p,k))] = env.db.ones()

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
        if TRACE: print 'op:',self
        env.binding[self.dst] = env.binding[self.src] * env.db.matrix(self.matmode,self.transpose)
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            if TRACE: print 'evalGrad',self.dst,'/',(p,k),'dict',env.binding.keys()
            if p==self.matmode.functor and k==self.matmode.arity:
                # df/dp r*M = (df/dp r) * M + r (df/dp M)
                #           = (df/dp r) * M + r I            if p==M
                #           = (df/dp r) * M + r I            else
                env.binding[Partial(self.dst,(p,k))] = \
                    env.binding[Partial(self.src,(p,k))] * env.db.matrix(self.matmode,self.transpose)  + env.binding[self.src] 
            else:
                env.binding[Partial(self.dst,(p,k))] = \
                    env.binding[Partial(self.src,(p,k))] * env.db.matrix(self.matmode,self.transpose)
                
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
        if TRACE: 
            print 'op:',self
        env.binding[self.dst] = env.binding[self.src].multiply( env.binding[self.src2] )
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            env.binding[Partial(self.dst,(p,k))] = \
                 env.binding[Partial(self.src,(p,k))].multiply( env.binding[self.src2] )  \
                 + env.binding[self.src].multiply( env.binding[Partial(self.src2,(p,k))] )

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
        env.binding[self.dst] =  env.binding[self.vec].multiply(env.binding[self.weighter].sum())
    def evalGrad(self,env):
        self.eval(env)
        for p,k in env.db.params:
            env.binding[Partial(self.dst,(p,k))] = \
                env.binding[Partial(self.vec,(p,k))].multiply(env.binding[self.weighter].sum()) \
                + env.binding[self.vec].multiply(env.binding[Partial(self.weighter,(p,k))].sum())
