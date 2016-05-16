# (C) William W. Cohen and Carnegie Mellon University, 2016

import logging
import traceback

import sys
import ops
import mutil
import tlerr

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
    def traceEvalCommencement(self):
        # call at beginning of eval
        if TRACE: print 'fn eval',self
    def traceEvalCompletion(self):
        # call at end of eval
        if TRACE: print 'fn eval /'+str(self)

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
        return [top] + reduce(lambda x,y:x+y, map(lambda o:[s for s in o.pprint(depth+1)], self.ops))
        #return [top] + map(lambda o:('| '*(depth+1))+o.pprint(), self.ops)
    def _doEval(self,db,values):
        self.traceEvalCommencement()
        #eval expression
        self.opEnv = ops.Envir(db)
        self.opEnv.bindList(self.opInputs,values)
        i=0
        for op in self.ops:
            i+=1
            #if TRACE: logging.debug("calling %d of %d %s from %s" % (i,len(self.ops),str(op),str(self)))
            op.eval(self.opEnv)
        self.result = self.opEnv[self.opOutput]
        #if TRACE: logging.debug("returning %s from %s" % (str(self.result.shape),str(self)))
        self.traceEvalCompletion()
        return self.result
    def backprop(self,delta,gradAccum):
        self.opEnv.delta[self.opOutput] = delta
        #if type(delta) != type(0) and delta.nnz == 0: raise tlerr.InvalidBackpropState("0 nonzero elements in delta")
        logging.debug("OpSeqFunction delta[%s] set to %s" % (self.opOutput,str(delta) if type(delta) == type(0) else mutil.summary(delta)))
        n = len(self.ops)
        #logging.debug("delta keys required: %s" % ",".join([self.ops[n-i-1].dst for i in range(n)]))
        for i in range(n):
            op = self.ops[n-i-1]
            logging.debug("delta key required: %s [%s]" % (op.dst,op.__class__.__name__))
            op.backprop(self.opEnv,gradAccum)
        assert len(self.opInputs)==1, 'bp for multiple input functions not implemented'
        logging.debug("I suppose deltas now set for %s" % ",".join([self.ops[n-i-1].src for i in range(n)]))
        return self.opEnv.delta[self.opInputs[0]]

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
        self.traceEvalCommencement()
        self.result = db.zeros(mutil.numRows(values[0]))
        self.traceEvalCompletion()
        return self.result
    def backprop(self,delta,gradAccum):
        return self.result


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
        self.traceEvalCommencement()
        addends = map(lambda f:f.eval(db,values), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        self.result = accum
        self.traceEvalCompletion()
        return self.result
    def backprop(self,delta,gradAccum):
        addends = map(lambda f:f.backprop(delta,gradAccum), self.funs)
        accum = addends[0]
        for i in range(1,len(addends)):
            accum = accum + addends[i]
        return accum

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
        self.traceEvalCommencement()
        unnorm = self.fun.eval(db,values)
        self.result = mutil.softmax(unnorm)
        self.traceEvalCompletion()
        return self.result
    def backprop(self,delta):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'

