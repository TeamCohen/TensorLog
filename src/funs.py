# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import numpy

import ops
import bcast


class ComputationNode(object):
    def __init__(self,fun,db,values,result,children=[]):
        self.children = children
        self.fun = fun
        self.db = db
        self.values = values
        self.result = result
        self.gradResult = {}
    def grad(self):
        for c in self.children:
            c.grad()
        self.fun.grad(self)
        return self.gradResult
    def show(self,depth=0):
        def about(m): return (type(m),m.data.shape)
        tab = '| '*(depth)
        print '%s %r' % (tab,self.fun)
        print '%s val %r:' % (tab,about(self.result))
        for w in self.gradResult:
            print '%s %s: %r' % (tab,w,about(self.gradResult[w]))
        for c in self.children:
            c.show(depth+1)


class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        return self.computationTree(db,values).result

    def evalGrad(self,db,values):
        return self.computationTree(db,values).grad()

    def computationTree(self,db,values):
        """When called with a MatrixDB and a list of input values v1,...,xk,
        executes some function f(v1,..,vk) and returns a computation
        tree that produces the output of the function.
        """
        assert False, 'abstract method called.'
    def gradResult(self,root,w):
        """Add derivatives to a computation tree.
        """
        assert False, 'abstract method called.'

class OpSeqFunction(Function):

    """A function defined by executing a sequence of operators."""
    def __init__(self,opInputs,opOutput,ops):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops

    def __repr__(self):
        shortOps = '[%r,...,%r]' % (self.ops[0],self.ops[-1])
        return 'OpSeqFunction(%r,%r,%r)' % (self.opInputs,self.opOutput,shortOps)

    def computationTree(self,db,values):
        #eval expression
        env = ops.Envir(db)
        env.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(env)
        return ComputationNode(self,db,values,env[self.opOutput])

    def grad(self,root):
        env = ops.Envir(root.db)
        env.bindList(self.opInputs,root.values)
        #initialize d(input)/d(param)=0 for each input/param pair
        registersForDerivsOfInputs = [ops.Partial(x,w) for x in self.opInputs for w in root.db.params]
        valuesForDerivsOfInputs = map(lambda r:root.db.ones() if r.f==r.x else root.db.zeros(), registersForDerivsOfInputs)
        env.bindList(registersForDerivsOfInputs,valuesForDerivsOfInputs)
        # execute ops with evalGrad
        for op in self.ops:
            op.evalGrad(env)
#        print 'ops',self.ops
#        print 'sample gradient:',env.register.keys()
        #rename the partial derivative from register are dopOutput/dParam, f=opOutput, x=param
        registersForDerivsOfOut = [ops.Partial(self.opOutput,w) for w in root.db.params]
        for r in registersForDerivsOfOut:
            root.gradResult[r.x] = env[r]
            m = root.db.matrixAsSymbolDict(env[r])
#            for r in m: print r,m[r].items()[0:3]
#            assert False

class NullFunction(OpSeqFunction):
    """Returns an all-zeros vector."""
    def __init__(self,lhsMode):
        self.opInputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
        self.opOutput = 'Y'
        self.ops = [ops.AssignZeroToVar(self.opOutput)]
    def __repr__(self):
        return 'NullFunction()'

class SumFunction(Function):
    """A function which computes the sum of a bunch of other functions."""
    def __init__(self,funs):
        self.funs = funs
    def __repr__(self):
        return 'SumFunction(%r)' % self.funs
    def computationTree(self,db,values):
        subtrees = map(lambda f:f.computationTree(db,values), self.funs)
        accum = subtrees[0].result
        for i in range(1,len(subtrees)):
            accum = accum + subtrees[i].result
        return ComputationNode(self,db,values,accum,subtrees)
    def grad(self,root):
        for w in root.db.params:
            accum = root.children[0].gradResult[w]
            for i in range(1,len(root.children)):
                accum = accum + root.children[i].gradResult[w]
            root.gradResult[w] = accum

class NormalizedFunction(Function):
    """A function which normalizes the result of another function."""
    def __init__(self,fun):
        self.fun = fun
    def __repr__(self):
        return 'NormalizedFunction(%r)' % self.fun
    def computationTree(self,db,values):
        subtree = self.fun.computationTree(db,values)
        result =  bcast.rowNormalize(subtree.result)
        return ComputationNode(self,db,values,result,children=[subtree])
    def grad(self,root):
        for subtree in root.children:
            subtree.fun.grad(subtree)
        # (f/g)' = (gf' - fg')/g^2
        def rowGrad(f,df):
            dg = df.sum()
            g = f.sum()
            return (g*df - f*dg)/(g*g)
        for w in root.db.params:
            f = root.children[0].result
            df = root.children[0].gradResult[w]
            f,df = bcast.broadcast2(f,df)
            numr = bcast.numRows(f)
            root.gradResult[w] = bcast.stack([rowGrad(f.getrow(i),df.getrow(i)) for i in range(numr)])


class StructuredLog(Function):
    def __init__(self,fun):
        self.fun = fun
    def __repr__(self):
        return 'StructuredLog(%r)' % self.fun
    def computationTree(self,db,values):
        subtree = self.fun.computationTree(db,values)
        P = subtree.result
        logP = SS.csr_matrix((numpy.log(P.data),P.indices,P.indptr),shape=P.get_shape())
        return ComputationNode(self,db,values,logP,children=[subtree])
    def grad(self,root):
        P = root.children[0].result
        oneByP = SS.csr_matrix(((1.0/P.data),P.indices,P.indptr),shape=P.get_shape())
        #TODO wrong?
        for w in root.db.params:        
            root.gradResult[w] = oneByP.multiply(root.children[0].gradResult[w])

class CrossEntropy(Function):
    def __init__(self,Y,fun):
        self.Y = Y
        self.fun = StructuredLog(fun)
    def computationTree(self,db,values):
        subtree = self.fun.computationTree(db,values)
        logP = subtree.result
        result = (-self.Y).multiply(logP)
        return ComputationNode(self,db,values,result,children=[subtree])
    def grad(self,root):
        logP = root.children[0].result
        for w in root.db.params:        
            root.gradResult[w] = (-self.Y).multiply(root.children[0].gradResult[w])

        


