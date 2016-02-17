# (C) William W. Cohen and Carnegie Mellon University, 2016

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B

#
# A slightly higher-level wrapper around Theano, for generality and
# convenience.  All the operations here can also be evaluated directly
# with scipy.
#

#
# misc interfaces to theano
#

def columnVectorParam(vec,name):
    (rows,cols) = vec.get_shape()
    assert cols==1,'not a column vector'
    return theano.shared(vec,name=name)

#
# environment - holds either computed values, or subexpressions
#

class Envir(object):
    """Holds a MatrixDB object and a group of variable bindings.
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
#
# functions
#

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
        return theano.function(inputs=inputs,outputs=outputs)
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
    def recurselyUse(self,pyfunction,db,values):
        """Binds variables named by self.inputs to values, in order, executes
        the inner op, and returns list of bindings for the output
        variables.
        """
        env = Envir(db)
        for i,v in enumerate(values):
            env.binding[self.inputs[i]] = v
        pyfunction(env)
        return [env.binding[y] for y in self.outputs]
    def eval(self,db,values):
        return self.recurselyUse(self.op.eval,db,values)
    def theanoExpr(self,db,subexprs):
        return self.recurselyUse(self.op.theanoExpr,db,subexprs)

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

#TODO do I need this, aside from for PropPR?
class WeightedSumFunction(Function):
    """Sum of a bunch of functions."""
    def __init__(self,weights,mainFuns,featureFuns):
        self.mainFuns = mainFuns
        self.featureFuns = featureFuns
        self.weights = weights
    def __str__(self):
        return "(" + " + ".join(map(lambda m,f:"%s*W^T*%s"%(m,f), self.mainFuns, self.featureFuns)) + ")"
    def __repr__(self):
        return "WeightedSumFunction("+",".join(map(repr,self.mainFuns))+";"+join(map(repr,self.featureFuns))+")"
    def recurselyUse(self,singleWeightedAddedFun,db,values):
        assert len(self.mainFuns)>0 and len(self.featureFuns)==len(self.mainFuns)
        baseValues = singleWeightedAddedFun(0,db,values)
        for i in range(1,len(self.mainFuns)):
            moreValues = singleWeightedAddedFun(i,db,values)
            assert len(moreValues)==len(baseValues)
            for j in range(len(moreValues)):
                baseValues[j] = baseValues[j] + moreValues[j]                
        return baseValues
    def eval(self,db,values):
        def weightedEval(i,db,values):
            unweighted = self.mainFuns[i].eval(db,values)
            features = self.featureFuns[i].eval(db,values)
            assert len(unweighted)==1 and len(features)==1,'not implemented'
            w = features[0].dot(self.weights.get_value())
            return [unweighted[0]*w[0,0]]
        return self.recurselyUse(weightedEval,db,values)
    def theanoExpr(self,db,values):
        def weightedExpression(i,db,values):
            unweighted = self.mainFuns[i].theanoExpr(db,values)
            features = self.featureFuns[i].theanoExpr(db,values)
            assert len(unweighted)==1 and len(features)==1,'not implemented'
            w = B.true_dot(features[0],self.weights)
            return [unweighted[0]*w[0,0]]
        return self.recurselyUse(weightedExpression,db,values)

#
# operators
#

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
        env.binding[self.dst] = env.db.matrixPreimage(self.mat)
    def theanoExpr(self,env):
        env.binding[self.dst] = env.db.matrixPreimage(self.mat)

#TODO rename AssignZerosToVar?

class ClearVar(Op):
    """Set the dst variable to an all-zeros row."""
    def __init__(self,dst):
        self.dst = dst
    def __str__(self):
        return "ClearVar(%s)" % (self.dst)
    def __repr__(self):
        return "ClearVar(%r)" % (self.dst)
    def eval(self,env):
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
        env.binding[self.dst] = env.db.onehot(self.onehotConst)
    def theanoExpr(self,env):
        c = env.db.onehot(self.onehotConst)
        theanoConstVec = S.CSR(c.data, c.indices, c.indptr, c.shape)
        env.binding[self.dst] = theanoConstVec

class ComponentwiseVecMulOp(Op):
    """ Computes dst = src*Diag(mat), i.e., the component-wise product of
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
        env.binding[self.dst] = env.binding[self.src].multiply( env.binding[self.src2] )
    def theanoExpr(self,env):
        env.binding[self.dst] = env.binding[self.src] * env.binding[self.src2]

#TODO used?

class ComponentwiseVecMulByOnehotOp(Op):
    """ Like ComponentwiseVecMulOp but one vector is a oneHot
    representation of a constant.
    """
    def __init__(self,dst,src,onehotConst):
        self.dst = dst
        self.src = src
        self.onehotConst = onehotConst
    def __str__(self):
        return "ComponentwiseVecMulByOnehotOp<%s = %s * %s>" % (self.dst,self.src,self.onehotConst)
    def __repr__(self):
        return "ComponentwiseVecMulByOnehotOp(%r,%r,%s)" % (self.dst,self.src,self.onehotConst)
    def eval(self,env):
        env.binding[self.dst] = env.binding[self.src].multiply( env.db.onehot(self.onehotConst) )
    def theanoExpr(self,env):
        c = env.db.onehot(self.onehotConst)
        theanoConstVec = S.CSR(c.data, c.indices, c.indptr, c.shape)
        env.binding[self.dst] = env.binding[self.src] * theanoConstVec

class WeightedOnehot(Op):
    def __init__(self,dst,weighter,onehotConst):
        self.dst = dst
        self.weighter = weighter
        self.onehotConst = onehotConst
    def __str__(self):
        return "WeightedOnehot<%s = %s * %s>" % (self.dst,self.weighter,self.onehotConst)
    def __repr__(self):
        return "WeightedOnehot<%s,%s,%s>" % (self.dst,self.weighter,self.onehotConst)
    def eval(self,env):
        env.binding[self.dst] = env.db.onehot(self.onehotConst) * env.binding[self.weighter].sum()
    def theanoExpr(self,env):
        #convert to a sparse vector constant
        c = env.db.onehot(self.onehotConst)
        theanoConstVec = S.CSR(c.data, c.indices, c.indptr, c.shape)
        env.binding[self.dst] = theanoConstVec * B.sp_sum(env.binding[self.weighter],sparse_grad=True)

class WeightedVec(Op):
    def __init__(self,dst,weighter,vec):
        self.dst = dst
        self.weighter = weighter
        self.vec = vec
    def __str__(self):
        return "WeightedVec<%s = %s * %s>" % (self.dst,self.weighter,self.vec)
    def __repr__(self):
        return "WeightedVec<%s,%s,%s>" % (self.dst,self.weighter,self.vec)
    def eval(self,env):
        env.binding[self.dst] = env.binding[self.vec] * env.binding[self.weighter].sum()
    def theanoExpr(self,env):
        env.binding[self.dst] = env.binding[self.vec] * B.sp_sum(env.binding[self.weighter],sparse_grad=True)

class VecMatMulOp(Op):
    """Op of the form "dst = src*mat"
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
        env.binding[self.dst] = env.binding[self.src] * env.db.matrix(self.matmode,self.transpose)
    def theanoExpr(self,env):
        env.binding[self.dst] = B.true_dot(env.binding[self.src], env.db.matrix(self.matmode,self.transpose))

