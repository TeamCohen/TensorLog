# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import numpy

import ops

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, take a list of input values as the inputs.
    """
    def eval(self,db,values):
        """When called with a MatrixDB and a list of input values v1,...,xk,
        executes some function f(v1,..,vk) and return the output of
        """
        assert False, 'abstract method called.'
    def evalGrad(self,db,values):
        """Return a dictionary mapping w=>the partial deriv of f wrt w for
        every param w, at the specified input values.
        """
        assert False, 'abstract method called.'

class OpSeqFunction(Function):
    """A function defined by executing a sequence of operators."""
    def __init__(self,opInputs,opOutput,ops):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops
    def eval(self,db,values):
        env = ops.Envir(db)
        env.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(env)
        return env[self.opOutput]
    def evalGrad(self,db,values):
        env = ops.Envir(db)
        env.bindList(self.opInputs,values)
        #initialize d(input)/d(param)=0 for each input/param pair
        derivsOfIns = [ops.Partial(x,w) for x in self.opInputs for w in db.params]
        env.bindList(derivsOfIns, [db.zeros()] * len(derivsOfIns))
        # execute ops with evalGrad
        for op in self.ops:
            op.evalGrad(env)
        #collect and rename the partial derivatives
        registersForDerivsOfOut = [ops.Partial(self.opOutput,w) for w in db.params]
        return dict([(r.f,env[r]) for r in registersForDerivsOfOut])

class NullFunction(OpSeqFunction):
    """Returns an all-zeros vector."""
    def __init__(self,lhsMode):
        self.opInputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
        self.opOutput = 'Y'
        self.ops = [ops.AssignZeroToVar(self.opOutput)]

class SumFunction(Function):
    """A function which computes the sum of a bunch of other functions."""
    def __init__(self,funs):
        self.funs = funs
    def eval(self,db,values):
        accum = self.funs[0].eval(db,values)
        for f in self.funs[1:]:
            accum = accum + f.eval(db,values)
        return accum
    def evalGrad(self,db,values):
        accumDict = self.funs[0].evalGrad(db,values)
        constZeros = db.zeros()
        for f in self.funs[1:]:
            deltaDict = f.evalGrad(db,values)
            for var,val in deltaDict.items():
                accumDict[var] = accumDict.get(var,constZeros) + deltaDict.get(var,constZeros)
        return accumDict

class NormalizedFunction(Function):

    """A function which normalizes the result of another function."""
    def __init__(self,fun):
        self.fun = fun

    def eval(self,db,values):
        m = self.fun.eval(db,values)
        numr = m.get_shape()[0]
        if numr==1:
            return m.multiply(1.0/m.sum())
        else:
            rows = [m.getrow(i) for i in range(numr)]
            return SS.vstack([r.multiply( 1.0/r.sum()) for r in rows], dtype='float64')

    def evalGrad(self,db,values):
        # (f/g)' = (gf' - fg')/g^2
        def gradrow(f,df):
            g = f.sum()
            dg = df.sum()
            return df.multiply(1.0/g) - f.multiply(dg/(g*g))
        m = self.fun.eval(db,values)
        dmDict = self.fun.evalGrad(db,values)
        gradDict = {}
        for w in db.params:
            dm = dmDict.get(w, db.zeros())
            numr = m.get_shape()[0]
            if numr==1:
                gradDict[w] = gradrow(m,dm)
            else:
                mrows = [m.getrow(i) for i in range(numr)]
                dmrows = [dm.getrow(i) for i in range(numr)]
                gradDict[w] = SS.vstack([gradrow(mrows[i],dmrows[i]) for i in range(numr)], dtype='float64')
        return gradDict
