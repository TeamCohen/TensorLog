# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# functions, which support evaluation and backprop
#

import sys
import logging
import copy

import opfunutil
import ops
import config
import mutil
import numpy

conf = config.Config()
conf.trace = False;         conf.help.trace =         "Print debug info during function eval"
conf.long_trace = False;    conf.help.long_trace =    "Print output of functions during eval - only for small tasks"

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    

    def eval(self,db,values,pad):
        self._checkDuplications()
        if conf.trace:
            print "Invoking:\n%s" % "\n. . ".join(self.pprint())
        pad[self.id].output = self._doEval(db,values,pad)
        if conf.trace:
            print "Function completed:\n%s" % "\n. . ".join(self.pprint())
            if conf.long_trace:
                for k,v in enumerate(values):
                    print '. input',k+1,':',db.matrixAsSymbolDict(values[k])
                print '. result :',db.matrixAsSymbolDict(result)
        return pad[self.id].output

    def backprop(self,delta,gradAccum,pad):
        if conf.trace:
            print "Backprop:\n%s" % "\n. . ".join(self.pprint())
        pad[self.id].delta = self._doBackprop(delta,gradAccum,pad)
        if conf.trace:
            print "Backprop completed:\n%s" % "\n. . ".join(self.pprint())
        return pad[self.id].delta

    def _checkDuplications(self):
        """For debugging/testing - poke around for copies of the same function
        somewhere in the tree."""
        kids = self.children()
        for i in range(len(kids)):
            for j in range(i+1,len(kids)):
                assert not kids[i] is kids[j], "Matching values not permitted for kids[%d],kids[%d]" % (i,j)
        for f in kids:
            if isinstance(f,Function):
                f._checkDuplications()

    def install(self,nextId=1):
        """ Give a numeric id to each function in this tree, and all the
        operations below it. """
        if hasattr(self,'id'): raise Exception("Tried to install already-installed function")
        self.id = nextId
        nextId += 1
        for f in self.children():
            #if hasattr(f,'id'):
            #    logging.debug("Deduping function %-2d %s" % (f.id,str(f)))
            #    f=self.dedupeChild(f) # TODO: replace with index
            nextId = f.install(nextId)
        return nextId

    def copy(self):
        """Return deep copy of function, to avoid duplication"""
        assert False, 'abstract method called'

    # these are used in pprint, and also in the debugging
    # visualization

    def pprint(self,depth=0):
        """Return list of lines in a pretty-print of the function.
        """
        top = ('%-2d ' % self.id) + self.pprintSummary()
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
    #def dedupeChild(self,f):
    #    assert False, 'abstract method called'

class OpSeqFunction(Function):
    """A function defined by executing a sequence of operators."""

    def __init__(self,opInputs,opOutput,ops,rule=None):
        self.opInputs = opInputs    #initial bindings to insert in Envir
        self.opOutput = opOutput  #finding bindings which indicate the output
        self.ops = ops
        self.rule = rule #recorded for debug/trace
    def __repr__(self):
        shortOps = '[%r,...,%r]' % (self.ops[0],self.ops[-1])
        return 'OpSeqFunction(%r,%r,%r)' % (self.opInputs,self.opOutput,shortOps)
    def pprintSummary(self):
        return '%s = OpSeqFunction(%r)' % (self.opOutput,self.opInputs)
    def pprintComment(self):
        return str(self.rule) if self.rule else '' 
    def _doEval(self,db,values,pad):
        #eval expression
        pad[self.id].opEnv = opfunutil.Envir(db)
        pad[self.id].opEnv.bindList(self.opInputs,values)
        for op in self.ops:
            op.eval(pad[self.id].opEnv,pad)
        return pad[self.id].opEnv[self.opOutput]
    def _doBackprop(self,delta,gradAccum,pad):
        pad[self.id].opEnv.delta[self.opOutput] = delta
        n = len(self.ops)
        for i in range(n):
            op = self.ops[n-i-1]
            op.backprop(pad[self.id].opEnv,gradAccum,pad)
        assert len(self.opInputs)==1, 'bp for multiple input functions not implemented'
        return pad[self.id].opEnv.delta[self.opInputs[0]]
    def children(self):
        return self.ops
    def copy(self):
        ret = OpSeqFunction(self.opInputs, self.opOutput, [o.copy() for o in self.ops], self.rule)
        return ret

class NullFunction(Function):
    """Returns an all-zeros vector."""

    def __init__(self,lhsMode):
        self.lhsMode = lhsMode
        self.opInputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
        self.opOutput = 'Y'
        self.ops = [ops.AssignZeroToVar(self.opOutput)]
    def __repr__(self):
        return 'NullFunction()'
    def pprintSummary(self):
        return 'NullFunction'
    def _doEval(self,db,values,pad):
        return db.zeros(mutil.numRows(values[0]))
    def _doBackprop(self,delta,gradAccum,pad):
        return pad[self.id].output
    def children(self):
        return []
    def copy(self):
        return NullFunction(self.lhsMode)

class LogFunction(Function):
    """Returns element-wise log of the output of the inner function."""

    def __init__(self,fun):
        self.fun = fun
    def __repr__(self):
        return 'LogFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'LogFunction'
    def _doEval(self,db,values,pad):
        self.inner = self.fun.eval(db,values,pad)
        return mutil.mapData(lambda d:numpy.log1p(d.clip(0,d)), self.inner)
    def _doBackprop(self,delta,gradAccum,pad):
        newDelta = mutil.mapData(lambda d:numpy.reciprocal(d+1), self.inner).multiply(delta)
        return self.fun.backprop(newDelta,gradAccum)
    def children(self):
        return [self.fun]
    def copy(self):
        ret = LogFunction(self.fun.copy())
        if hasattr(self,'inner'): ret.inner = self.inner
        return ret

class SumFunction(Function):
    """A function which computes the sum of a bunch of other functions."""

    def __init__(self,funs):
        self.funs = funs
    def __repr__(self):
        return 'SumFunction(%r)' % self.funs
    def pprintSummary(self):
        return 'SumFunction'
    def _doEval(self,db,values,pad):
        addends = map(lambda f:f.eval(db,values,pad), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        return accum
    def _doBackprop(self,delta,gradAccum,pad):
        addends = map(lambda f:f.backprop(delta,gradAccum,pad), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            try:
                accum = accum + addends[i]
            except:
                print "accum %s" % mutil.summary(accum)
                print "addends[%d] %s" % (i,mutil.summary(addends[i]))
                print "\n".join(self.pprint())
                raise
        return accum
    def children(self):
        return self.funs
    def copy(self):
        return SumFunction([f.copy() for f in self.funs])

class SoftmaxFunction(Function):
    """A function which computes row-wise softmax of an inner function."""

    def __init__(self,fun):
        self.fun = fun
    def __repr__(self):
        return 'SoftmaxFunction(%r)' % self.fun
    def pprintSummary(self):
        return 'SoftmaxFunction'
    def _doEval(self,db,values,pad):
        unnorm = self.fun.eval(db,values,pad)
        return mutil.softmax(db,unnorm)
    def _doBackprop(self,delta,pad):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'
    def children(self):
        return [self.fun]
    def copy(self):
        return SoftmaxFunction(self.fun.copy())
