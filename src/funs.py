# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import numpy as np
import logging

import ops
import bcast

# TODO use scipy.misc.logsumexp
# then: log of softmax is: outputs - logsumexp(outputs, axis=1, keepdims=True)
# see https://github.com/HIPS/autograd/blob/master/examples/neural_net.py

#TODO: how to associated the backprop nodes with the computation node?
#mybe subclass ComputationNode with forward and backprop classes?  but
#then do I really need a computation tree class at all?

def trace(): return logging.getLogger().isEnabledFor(logging.DEBUG)

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        assert False, 'abstract method called'
    def pprint(self,depth=0):
        """Return list of lines in a pretty-print of the function.
        """
        assert False, 'abstract method called'

class OpSeqFunction(Function):

    """A function defined by executing a sequence of operators."""
    def __init__(self,opInputs,opOutput,ops):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops
        self.result = None
        self.opEnv = None #caches environment to evaluate the ops
    def __repr__(self):
        shortOps = '[%r,...,%r]' % (self.ops[0],self.ops[-1])
        return 'OpSeqFunction(%r,%r,%r)' % (self.opInputs,self.opOutput,shortOps)
    def pprint(self,depth=0):
        return [('| '*depth) + 'OpSeqFunction:'] + map(lambda o:('| '*(depth+1))+repr(o), self.ops)
    def eval(self,db,values):
        #eval expression
        self.opEnv = ops.Envir(db)
        self.opEnv.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(self.opEnv)
        self.result = self.opEnv[self.opOutput]
        return self.result
    def backprop(self,delta,updates):
        self.opEnv.delta[self.opOutput] = delta
        n = len(self.ops)
        for i in range(n):
            op = self.ops[n-i-1]
            #print 'fun bp op',n-i-1,op
            op.backprop(self.opEnv,updates)

class NullFunction(OpSeqFunction):
    """Returns an all-zeros vector."""
    def __init__(self,lhsMode):
        self.opInputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
        self.opOutput = 'Y'
        self.ops = [ops.AssignZeroToVar(self.opOutput)]
    def __repr__(self):
        return 'NullFunction()'
    def pprint(self,depth=0):
        return [('| '*depth) + repr(self)]
    def eval(self,db,values):
        self.result = db.zeros()
        return self.result
    def backprop(self,delta,updates):
        pass

class SumFunction(Function):
    """A function which computes the sum of a bunch of other functions."""
    def __init__(self,funs):
        self.funs = funs
        self.result = None
    def __repr__(self):
        return 'SumFunction(%r)' % self.funs
    def pprint(self,depth=0):
        return [('| '*depth) + 'SumFunction:'] + reduce(lambda x,y:x+y, map(lambda f:f.pprint(depth=depth+1), self.funs))
    def eval(self,db,values):
        addends = map(lambda f:f.eval(db,values), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        self.result = accum
        return self.result
    def backprop(self,delta,updates):
        for f in self.funs:
            f.backprop(delta,updates)

class SoftmaxFunction(Function):
    """A function which computes row-wise softmax."""
    def __init__(self,fun):
        self.fun = fun
        self.result = None
    def __repr__(self):
        return 'SoftmaxFunction(%r)' % self.fun
    def pprint(self,depth=0):
        return [('| '*depth) + 'SoftmaxFunction:'] + self.fun.pprint(depth=depth+1)
    def eval(self,db,values):
        unnorm = self.fun.eval(db,values)
        self.result = bcast.softmax(unnorm)
        return self.result
    def backprop(self,delta):
        assert False, 'should not call this directly'

#TODO modify this - we'll need it for tracking loss 

#class CrossEntropy(Function):
#    def __init__(self,Y,fun):
#        self.Y = Y
#        self.fun = StructuredLog(fun)
#    def computationTree(self,db,values):
#        subtree = self.fun.computationTree(db,values)
#        logP = subtree.result
#        result = (-self.Y).multiply(logP)
#        return ComputationNode(self,db,values,result,children=[subtree])
#    def grad(self,root):
#        logP = root.children[0].result
#        for w in root.db.params:        
#            root.gradResult[w] = (-self.Y).multiply(root.children[0].gradResult[w])

