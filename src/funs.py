# (C) William W. Cohen and Carnegie Mellon University, 2016

import logging
import traceback

import sys
import ops
import mutil

TRACE = False

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        result = self._doEval(db,values)
        if TRACE:
            print "Function completed:\n%s" % "\n. . ".join(self.pprint())
            for k,v in enumerate(values):
                print '. input',k+1,':',db.matrixAsSymbolDict(values[k])
            print '. result :',db.matrixAsSymbolDict(result)
        return result
    def pprint(self,depth=0):
        """Return list of lines in a pretty-print of the function.
        """
        assert False, 'abstract method called'

class OpSeqFunction(Function):

    """A function defined by executing a sequence of operators."""
    def __init__(self,opInputs,opOutput,ops,rule=None):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops
        self.result = None
        self.rule = rule #recorded for debug/trace
        self.opEnv = None #caches environment to evaluate the ops
    def __repr__(self):
        shortOps = '[%r,...,%r]' % (self.ops[0],self.ops[-1])
        return 'OpSeqFunction(%r,%r,%r)' % (self.opInputs,self.opOutput,shortOps)
    def pprint(self,depth=0):
        top = ('| '*depth) + '%s = OpSeqFunction(%r):' % (self.opOutput,self.opInputs)
        if self.rule: top = top + '\t\t// ' + str(self.rule)
        return [top] + map(lambda o:('| '*(depth+1))+o.pprint(), self.ops)
    def _doEval(self,db,values):
        #eval expression
        self.opEnv = ops.Envir(db)
        self.opEnv.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(self.opEnv)
        self.result = self.opEnv[self.opOutput]
        return self.result
    def backprop(self,delta,gradAccum):
        self.opEnv.delta[self.opOutput] = delta
        n = len(self.ops)
        for i in range(n):
            op = self.ops[n-i-1]
            op.backprop(self.opEnv,gradAccum)

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
    def _doEval(self,db,values):
        self.result = db.zeros(mutil.numRows(values[0]))
        return self.result
    def backprop(self,delta,gradAccum):
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
    def _doEval(self,db,values):
        addends = map(lambda f:f.eval(db,values), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            assert accum.get_shape()==addends[i].get_shape(), \
                'shape error %r vs %r for addend %d\n%s' % (accum.get_shape(),addends[i].get_shape(),i,("\n > ".join(self.pprint())))
            accum = accum + addends[i]
        self.result = accum
        return self.result
    def backprop(self,delta,gradAccum):
        for f in self.funs:
            f.backprop(delta,gradAccum)

class SoftmaxFunction(Function):
    """A function which computes row-wise softmax."""
    def __init__(self,fun):
        self.fun = fun
        self.result = None
    def __repr__(self):
        return 'SoftmaxFunction(%r)' % self.fun
    def pprint(self,depth=0):
        return [('| '*depth) + 'SoftmaxFunction:'] + self.fun.pprint(depth=depth+1)
    def _doEval(self,db,values):
        unnorm = self.fun.eval(db,values)
        self.result = mutil.softmax(unnorm)
        return self.result
    def backprop(self,delta):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'

