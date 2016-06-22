# (C) William W. Cohen and Carnegie Mellon University, 2016

import logging
import traceback

import sys
import ops
import mutil
import config
import numpy

conf = config.Config()
conf.trace = False;         conf.help.trace =         "Print debug info during function eval"
conf.long_trace = False;    conf.help.long_trace =    "Print output of functions during eval - only for small tasks"

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        if conf.trace:
            print "Invoking:\n%s" % "\n. . ".join(self.pprint())
        result = self._doEval(db,values)
        if conf.trace:
            print "Function completed:\n%s" % "\n. . ".join(self.pprint())
            if conf.long_trace:
                for k,v in enumerate(values):
                    print '. input',k+1,':',db.matrixAsSymbolDict(values[k])
                print '. result :',db.matrixAsSymbolDict(result)
        return result
    def backprop(self,delta,gradAccum):
        if conf.trace:
            print "Backprop:\n%s" % "\n. . ".join(self.pprint())
        result = self._doBackprop(delta,gradAccum)
        if conf.trace:
            print "Backprop completed:\n%s" % "\n. . ".join(self.pprint())
        return result
    # these are used in pprint, and also in the debugging
    # visualization
    def pprint(self):
        """Return list of lines in a pretty-print of the function.
        """
        top = self.pprintSummary()
        comment = self.pprintComment()
        result = [top + ' # ' + comment] if comment else [top]
        for c in self.children():
            for s in c.pprint():
                result.append('| ' + s)
        return result
    def pprintSummary(self):
        assert False, 'abstract method called'
    def pprintComment(self):
        return ''
    def children(self):
        """Return list of child functions, for visualization"""
        assert False, 'abstract method called'

class OpSeqFunction(Function):

    """A function defined by executing a sequence of operators."""
    def __init__(self,opInputs,opOutput,ops,rule=None):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops
        self.output = None
        self.rule = rule #recorded for debug/trace
        self.opEnv = None #caches environment to evaluate the ops
    def __repr__(self):
        shortOps = '[%r,...,%r]' % (self.ops[0],self.ops[-1])
        return 'OpSeqFunction(%r,%r,%r)' % (self.opInputs,self.opOutput,shortOps)
    def pprintSummary(self):
        return '%s = OpSeqFunction(%r)' % (self.opOutput,self.opInputs)
    def pprintComment(self):
        return str(self.rule) if self.rule else '' 
    def _doEval(self,db,values):
        #eval expression
        self.opEnv = ops.Envir(db)
        self.opEnv.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(self.opEnv)
        self.output = self.opEnv[self.opOutput]
        return self.output
    def _doBackprop(self,delta,gradAccum):
        self.opEnv.delta[self.opOutput] = delta
        n = len(self.ops)
        for i in range(n):
            op = self.ops[n-i-1]
            op.backprop(self.opEnv,gradAccum)
        assert len(self.opInputs)==1, 'bp for multiple input functions not implemented'
        return self.opEnv.delta[self.opInputs[0]]
    def children(self):
        return self.ops

class NullFunction(Function):
    """Returns an all-zeros vector."""
    def __init__(self,lhsMode):
        self.opInputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
        self.opOutput = 'Y'
        self.ops = [ops.AssignZeroToVar(self.opOutput)]
    def __repr__(self):
        return 'NullFunction()'
    def pprintSummary(self):
        return 'NullFunction'
    def _doEval(self,db,values):
        self.output = db.zeros(mutil.numRows(values[0]))
        return self.output
    def _doBackprop(self,delta,gradAccum):
        return self.output
    def children(self):
        return []

class LogFunction(Function):
    """Returns an all-zeros vector."""
    def __init__(self,fun):
        self.fun = fun
        self.output = None
    def __repr__(self):
        return 'LogFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'LogFunction'
    def _doEval(self,db,values):
        self.inner = self.fun.eval(db,values)
        self.output = mutil.mapData(lambda d:numpy.log1p(d.clip(0,d)), self.inner)
        #self.output = self.inner
        return self.output
    def _doBackprop(self,delta,gradAccum):
        newDelta = mutil.mapData(lambda d:numpy.reciprocal(d+1), self.inner).multiply(delta)
        return self.fun.backprop(newDelta,gradAccum)
    def children(self):
        return [self.fun]

class SumFunction(Function):
    """A function which computes the sum of a bunch of other functions."""
    def __init__(self,funs):
        self.funs = funs
        self.output = None
    def __repr__(self):
        return 'SumFunction(%r)' % self.funs
    def pprintSummary(self):
        return 'SumFunction'
    def _doEval(self,db,values):
        addends = map(lambda f:f.eval(db,values), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        self.output = accum
        return self.output
    def _doBackprop(self,delta,gradAccum):
        addends = map(lambda f:f.backprop(delta,gradAccum), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        return accum
    def children(self):
        return self.funs

class SoftmaxFunction(Function):
    """A function which computes row-wise softmax."""
    def __init__(self,fun):
        self.fun = fun
        self.output = None
    def __repr__(self):
        return 'SoftmaxFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'SoftmaxFunction'
    def _doEval(self,db,values):
        unnorm = self.fun.eval(db,values)
        self.output = mutil.softmax(db,unnorm)
        return self.output
    def _doBackprop(self,delta):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'
    def children(self):
        return [self.fun]


