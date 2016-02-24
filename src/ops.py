# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse
import numpy

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B

TRACE=True

#
# A slightly higher-level wrapper around Theano, for generality and
# convenience.  All the operations here can also be evaluated directly
# with scipy so theano is less than 100% necessary.  
#
# This is also the only module in tensorlog that should directly use
# theano.  (If I was being less lazy there would be a subclass of this
# which includes the theano stuff and a factory and such for
# operators.)

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
    def theanoExpr(self,db,subexprs):
        """Return a theano expression that computes the output from a list of
        subexpressions."""
        assert False, 'abstract method called.'
    def theanoPredictFunction(self,db,symbols):
        """A theano.function that implements the same function as the eval
        routine."""
        inputs = map(lambda s:S.csr_matrix(s), symbols)
        outputs = self.theanoExpr(db,inputs)
        return theano.function(inputs=inputs,outputs=outputs,on_unused_input='ignore')
    def recurselyUse(self,pyfunction,db,values):
        """Implements either eval, if pyfunction is self.op.eval and the
        values are inputs, or theanoExpr, if pyfunction is
        self.op.theanoExpr, if the 'values' are subexpressions.  In
        eval mode, it bBinds input variables to values, in order,
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
    #TODO combine these
    def eval(self,db,values):
        env = Envir(db)
        for i,v in enumerate(values):
            env.binding[self.inputs[i]] = v
        self.op.eval(env)
        return [env.binding[y] for y in self.outputs]
    def theanoExpr(self,db,subexprs):
        env = Envir(db)
        for i,v in enumerate(subexprs):
            env.binding[self.inputs[i]] = v
        self.op.theanoExpr(env)
        return [env.binding[y] for y in self.outputs]

class SumFunction(Function):
    """Sum of a bunch of functions."""
    def __init__(self,funs):
        self.funs = funs
    def __str__(self):
        return "(" + " + ".join(map(repr,self.funs)) + ")"
    def __repr__(self):
        return "SumFunction("+repr(self.funs)+")"
    def recurselyUse(self,pyfunctions,db,values):
        """Pyfunctions is list of python functions, not one as in other
        instances.  Add up the results of these applying functions on
        the list of values and return the result.
        """
        assert len(pyfunctions)>1
        baseValues = pyfunctions[0](db,values)
        for f in pyfunctions[1:]:
            moreValues = f(db,values)
            assert len(moreValues)==len(baseValues)
            for j in range(len(moreValues)):
                #warning: when used to produce a theanoExpr this
                #assumes we can use the '+' operator
                baseValues[j] = baseValues[j] + moreValues[j]
        return baseValues
    def eval(self,db,values):
        pyfuns = [fun.eval for fun in self.funs]
        return self.recurselyUse(pyfuns,db,values)
    def theanoExpr(self,db,values):
        pyfuns = [fun.theanoExpr for fun in self.funs]
        return self.recurselyUse(pyfuns,db,values)

##############################################################################
#
# parameter database - the regular database will delegate to this so
# that matrices can be 'parameters', if necessary.
# 
##############################################################################

class ParameterDB(object):
    def __init__(self):
        self.paramEncoding = {}
        self.arity = {}

    def insert(self,mat,predicateFunctor,predicateArity):
        theanoShared = theano.shared(mat,name=predicateFunctor)
        self.paramEncoding[predicateFunctor] = theanoShared
        self.arity[predicateFunctor] = predicateArity
        return theanoShared

    def matrix(self,mode,transpose=False,expressionContext=False):
        if not mode.functor in self.paramEncoding: return None
        else: 
            assert self.arity[mode.functor]==2
            if matrixdb.MatrixDB.transposeNeeded(mode,transpose):
                if expressionContext:
                    return self.paramEncoding[mode.functor].transpose()
                else:
                    return self.paramEncoding[mode.functor].get_value().transpose()            
            else:
                if expressionContext:
                    return self.paramEncoding[mode.functor]
                else:
                    return self.paramEncoding[mode.functor].get_value()

    def matrixPreimage(self,mode,expressionContext=False):
        if not mode.functor in self.paramEncoding: return None
        else: 
            assert self.arity[mode.functor]==1,'p(X,Y) for unbound or constant Y not implemented for parameters'
            if expressionContext:
                return self.paramEncoding[mode.functor]
            else:
                return self.paramEncoding[mode.functor].get_value()


##############################################################################
#
# operators
#
##############################################################################

class Op(object):
    """Like a function but side-effects an environment.  More
    specifically, this is the tensorlog encoding for matrix-db
    'operations' which can be 'eval'ed, or implemented by a theano
    subexpression.  Operations typically specify src and dst variable
    names and eval-ing them will side-effect an environment, by
    binding the src to some function of the dst's binding.
    """
    def eval(self,env):
        assert False,'abstract method called'
    def theanoExpr(self,env):
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
    def theanoExpr(self,env):
        for op in self.ops:
            op.theanoExpr(env)

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
    def modifyEnvironment(self,pyfunction,env):
        vals = [env.binding[self.src]]
        outputs = pyfunction(self.tensorlogProg.db, vals)
        env.binding[self.dst] = outputs[0]
    def eval(self,env):
        subfun = self.tensorlogProg.function[(self.mode,self.depth)]
        self.modifyEnvironment(subfun.eval,env)
    def theanoExpr(self,env):
        subfun = self.tensorlogProg.function[(self.mode,self.depth)]
        self.modifyEnvironment(subfun.theanoExpr,env)

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
    def theanoExpr(self,env):
        env.binding[self.dst] = env.db.matrixPreimage(self.mat,expressionContext=True)


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
    def theanoExpr(self,env):
        env.binding[self.dst] = env.db.zeros()

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
    def theanoExpr(self,env):
        c = env.db.onehot(self.onehotConst)
        theanoConstVec = S.CSR(c.data, c.indices, c.indptr, c.shape)
        env.binding[self.dst] = theanoConstVec

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
    def theanoExpr(self,env):
        env.binding[self.dst] = B.true_dot(env.binding[self.src], env.db.matrix(self.matmode,self.transpose,expressionContext=True))

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
    def theanoExpr(self,env):
        env.binding[self.dst] = env.binding[self.src] * env.binding[self.src2] 

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
    def theanoExpr(self,env):
        env.binding[self.dst] = env.binding[self.vec] * B.sp_sum(env.binding[self.weighter],sparse_grad=True)

