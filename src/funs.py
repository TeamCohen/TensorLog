# (C) William W. Cohen and Carnegie Mellon University, 2016

import logging
import traceback

import sys
import ops
import mutil
import tlerr

TRACE = True
LONG_TRACE = False
STRICT=True

class Function(object):
    """The tensorlog representation of a function. This supports eval and
    evalGrad operations, and take a list of input values as the inputs.
    """
    def eval(self,db,values):
        if TRACE:
            print "Invoking:\n%s" % "\n. . ".join(self.pprint())
        result = self._doEval(db,values)
        if LONG_TRACE:
            print "Function completed:\n%s" % "\n. . ".join(self.pprint())
            if LONG_TRACE:
                for k,v in enumerate(values):
                    print '. input',k+1,':',db.matrixAsSymbolDict(values[k])
                print '. result :',db.matrixAsSymbolDict(result)
        return result
    def backprop(self,delta,gradAccum):
        if TRACE:
            print "Backprop:\n%s" % "\n. . ".join(self.pprint())
        result = self._doBackprop(delta,gradAccum)
        if TRACE:
            print "Backprop completed:\n%s" % "\n. . ".join(self.pprint())
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
    def _doBackprop(self,delta,gradAccum):
        self.opEnv.delta[self.opOutput] = delta
        #if type(delta) != type(0) and delta.nnz == 0: raise tlerr.InvalidBackpropState("0 nonzero elements in delta")
        if TRACE: print("OpSeqFunction(%s,%s) delta[%s] set to %s" % (self.opInputs,self.opOutput,self.opOutput,str(delta) if type(delta) == type(0) else mutil.summary(delta)))
        if TRACE: print("opEnv.delta now %s" % (str(self.opEnv.delta)))
        if TRACE: print("opEnv now %s" % (str(self.opEnv.register)))
        n = len(self.ops)
        #logging.debug("delta keys required: %s" % ",".join([self.ops[n-i-1].dst for i in range(n)]))
        for i in range(n):
            op = self.ops[n-i-1]
            if TRACE: print("OpSeqFunction delta key required: %s [%s]" % (op.dst,op.__class__.__name__))
            #print 'calling backprop on op',n-i-1,str(op)
            op.backprop(self.opEnv,gradAccum)
            if TRACE: 
                for (functor,arity),delta in gradAccum.items():
                    print("OpSeqFunction(%s,%s) gradAccum for %s after %s: %s" % (self.opInputs,self.opOutput,functor,str(op),mutil.summary(delta)))
                    if STRICT and delta.min() < -1e5: raise tlerr.InvalidBackpropState("bad gradAccum delta at %s" % self)
            #print 'op.backprop',n-i-1,'finished'
        assert len(self.opInputs)==1, 'bp for multiple input functions not implemented'
        if TRACE: print("deltas should now be set for %s" % ",".join([self.ops[n-i-1].src for i in range(n)]))
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
        #self.traceEvalCommencement()
        self.result = db.zeros(mutil.numRows(values[0]))
        #self.traceEvalCompletion()
        return self.result
    def _doBackprop(self,delta,gradAccum):
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
    def _doBackprop(self,delta,gradAccum):
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
        unnorm = self.fun.eval(db,values)
        self.result = mutil.softmax(db,unnorm)
        return self.result
    def _doBackprop(self,delta):
        # see comments for learner.crossEntropyGrad
        assert False, 'should not call this directly'

