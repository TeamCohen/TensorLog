# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# functions, which support evaluation and backprop
#

import sys
import logging
import traceback

import ops
import config
import mutil
import putil
import numpy

#experimentally the parallel_sum option doesn't seem to help
#time-wise: CPU usage rarely gets over 150% it also causes errors in
#cora-linear-expt.

conf = config.Config()
conf.trace = False;         conf.help.trace =         "Print debug info during function eval"
conf.long_trace = False;    conf.help.long_trace =    "Print output of functions during eval - only for small tasks"
conf.parallel_sum = False;   conf.help.parallel_sum =  "Evaluate sum operations in a parallel, one thread per addend."

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        if conf.trace:
            print "Invoking:\n%s" % "\n. . ".join(self.pprint())
        self.output = self._doEval(db,values)
        if conf.trace:
            print "Function completed:\n%s" % "\n. . ".join(self.pprint())
            if conf.long_trace:
                for k,v in enumerate(values):
                    print '. input',k+1,':',db.matrixAsSymbolDict(values[k])
                print '. result :',db.matrixAsSymbolDict(result)
        return self.output
    def backprop(self,delta,gradAccum):
        if conf.trace:
            print "Backprop:\n%s" % "\n. . ".join(self.pprint())
        self.delta = self._doBackprop(delta,gradAccum)
        if conf.trace:
            print "Backprop completed:\n%s" % "\n. . ".join(self.pprint())
        return self.delta
    # these are used in pprint, and also in the debugging
    # visualization
    def pprint(self,depth=0):
        """Return list of lines in a pretty-print of the function.
        """
        top = self.pprintSummary()
        comment = self.pprintComment()
        result = [top + ' # ' + comment] if comment else [top]
        for c in self.children():
            for s in c.pprint(depth):
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
        self.delta = None
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
        i=0
        for op in self.ops:
            i+=1
            #if TRACE: logging.debug("calling %d of %d %s from %s" % (i,len(self.ops),str(op),str(self)))
            op.eval(self.opEnv)
        return self.opEnv[self.opOutput]
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
        self.output = None
        self.delta = None
    def __repr__(self):
        return 'NullFunction()'
    def pprintSummary(self):
        return 'NullFunction'
    def _doEval(self,db,values):
        return db.zeros(mutil.numRows(values[0]))
    def _doBackprop(self,delta,gradAccum):
        return self.output
    def children(self):
        return []

class LogFunction(Function):
    """Returns an all-zeros vector."""
    def __init__(self,fun):
        self.fun = fun
        self.output = None
        self.delta = None
    def __repr__(self):
        return 'LogFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'LogFunction'
    def _doEval(self,db,values):
        self.inner = self.fun.eval(db,values)
        return mutil.mapData(lambda d:numpy.log1p(d.clip(0,d)), self.inner)
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
        self.delta = None
    def __repr__(self):
        return 'SumFunction(%r)' % self.funs
    def pprintSummary(self):
        return 'SumFunction'
    def _doEval(self,db,values):
        mapfun = putil.multithreaded_map if conf.parallel_sum else map
        addends = mapfun(lambda f:f.eval(db,values), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        return accum
    def _doBackprop(self,delta,gradAccum):
        mapfun = putil.multithreaded_map if conf.parallel_sum else map
        addends = mapfun(lambda f:f.backprop(delta,gradAccum), self.funs)
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
        self.delta = None
    def __repr__(self):
        return 'SoftmaxFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'SoftmaxFunction'
    def _doEval(self,db,values):
        unnorm = self.fun.eval(db,values)
        return mutil.softmax(db,unnorm)
    def _doBackprop(self,delta):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'
    def children(self):
        return [self.fun]


