# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import collections
import logging
import getopt

import wamcompiler
import wamplugin
import parser

import util
import sil
import symtab

DEFAULT_DEBUG_MODE = True  #keep variable names around for debugging?
MAXDEPTH = 10              #default max depth for allSolutionsDFS
DEFAULT_TRACE = False      #default trace-output flag for allSolutionsDFS
#DEFAULT_TRACE = True
#logging.basicConfig(level=logging.DEBUG)

class Interpreter(object):

    class CallStackFrame(object):
        """Enough information to 'return' to a prior state."""
        def __init__(self,state):
            self.hp = len(state.heap)
            self.rp = len(state.registers)
            self.pc = state.pc
            self.jumpTo = state.jumpTo
        def __hash__(self):
            return hash(self.hp)^hash(self.rp)^hash(self.pc)^hash(self.jumpTo)
        def __eq__(self,other):
            return self.hp==other.hp and self.rp==other.rp and self.pc==other.pc and self.jumpTo==other.jumpTo
        def __str__(self):
            return "sf:" + str([self.hp,self.rp,self.pc,self.jumpTo])

    class AbstractState(object):
        """State of the interpreter.  States are stored and retrieved to allow
        for backtracking, and to build a proof graph.  You can call

        savedState = wamInterp.state.save()
        
        to save an interpreter state, and 

        wamInterp.state = State.restore(savedState)

        to restore one. Saved states are immutable and hashable.
        """

        ####################
        #
        # Data structures used:
        
        # The 'heap' is a list that stores arguments being passed
        # to/from predicates, and is organized as a stack, with 'hp'
        # pointing to the next free element. Elements on the heap can
        # be constants or variables.  See comments near 'heap
        # utilities', below.

        # 'Registers' is a list that stores local variables associated
        # with a clause, and is also organized as a stack. Each
        # register holds an index into the heap.

        # 'varNameList' is a list of meaningful variable names for the local variables in
        # the stack - only maintained if debugMode==True

        # 'calls' is a stack of CallStackFrame's

        # 'pc' is the program counter, and 'jumpTo' is a label
        # defining where to jump to.  Together these define the next
        # step for the abstract machine to take.

        # 'completed' is a flag - did the computation finish?
        # 'failed' is a flag - did the computation fail?

        ####################
        #
        # Below: Heap utilities that depend only on the state.

        # The heap contains constants and variables.  Bound variables
        # point to another position (index) on the heap, and free
        # variables point to their own index.  Constants are stored as
        # negative integers, which are decoded using the constantTable
        # associated with the Interpreter instance.  (The
        # corresponding positive int is the id in the constantTable,
        # which has positive int id's.
        # 
        # Every variable must point to a LOWER index on the heap: eg
        # the variable at index 7 can point to 5 but not vice versa.
        # So, no reference loops are allowed.  If you follow a chain
        # of references you should always reach a free variable, or a
        # constant.  Use deref() to follow a chain; my convention is
        # to use ri for deref(i).  Hence heap position ri is the
        # "prototype" for all the variables in the chain rooted at i;
        # changing ri will change all of them, and in unification, you
        # should ONLY change these prototype cells, never any
        # intermediate cells in a change.
        #
        # In unification, you are only allowed to turn a free variable
        # (ie the end of a reference chain) into a constant.
        #
        # Once you've found out that a chain starting at i finally
        # leads to index ri, calling collapsePointers(i,ri) will make
        # future chain-following faster.  It should not change the
        # semantics.
        #
        #
        # The has*, get*, create*, and copy* things are all functions,
        # so as to support other heap implementations, eg storing the
        # constant strings explicitly in the heap.

        def hasConstAtI(self,i): 
            """True iff there is a constant at heap position i."""
            return self.heap[i]<0

        def hasVarAtI(self,i): 
            """True iff there is a variable at heap position i."""
            return self.heap[i]>=0

        def hasFreeVarAtI(self,i): 
            """True iff there is an unbound variable at heap position i."""
            return self.heap[i]==i

        def getVarValueAtI(self,i): 
            """Get the value of the variable stored at this heap position."""
            return self.heap[i]

        def getIdOfConstValueAtI(self,i): 
            """Get the id, in the Interpreter's constantTable, of the constant
            stored in this heap position."""
            assert self.heap[i]<0
            return -self.heap[i]

        def createConstCellById(self,id):
            """Create a heap cell that stores a constant with the given id"""
            assert id>=1
            return -id

        def createVarCell(self,a):
            """Create a heap cell that stores a variable bound to heap position a"""
            return a

        def copyConstCell(self,i):
            """Create a copy of the constant cell at position i."""
            return self.heap[i]

        #
        # Variable referencing
        #

        def deref(self,heapIndex):
            """Dereference a variable, ie, follow pointers till you reach an
            unbound variable or a constant."""
            startIndex = heapIndex
            while (not self.hasConstAtI(heapIndex) and not(self.hasFreeVarAtI(heapIndex))):
                heapIndex = self.getVarValueAtI(heapIndex)
            return heapIndex
        
        def __str__(self):
            callsString = "[" + ",".join(map(str,self.calls)) + "]"
            stat = ''
            if self.completed: stat += '*'
            if self.failed: stat += '!'
            params = " ".join(map(str,[self.heap,self.registers,callsString,self.pc,self.jumpTo]))
            vars = ("; "+ ",".join(self.varNameList)) if self.varNameList else ""
            return "state<" + params + vars + ">" + stat

    class MutableState(AbstractState):
        """An AbstractState that can be modified.  This is the representation
        of interpreter state that is used in execution."""

        def __init__(self):
            self.heap = []
            self.registers = []
            self.varNameList = []
            self.calls = []
            self.pc = -1
            self.jumpTo = None
            self.completed = self.failed = False

        def immutableVersion(self):
            """ Immutable, hashable version of this state."""
            return Interpreter.ImmutableState(self)

        def collapsePointers(self,i,finalIndex):
            """Given index i, the start of a chain of heap variables that ends at
            index finalIndex, make all the variables in that chain
            point directly to finalIndex."""
            while i!=finalIndex and self.hasVarAtI(i):
                nextI = self.getVarValueAtI(i)
                assert finalIndex<i, 'bad collapsePointers from %d to %d: %s' % (i,finalIndex,str(self))
                self.heap[i] = self.createVarCell(finalIndex)
                i = nextI

    class ImmutableState(AbstractState):
        """An immutable, hashable version of an interpreter state."""

        def __init__(self,mutableState):
            self.flagObjects = (mutableState.pc,mutableState.jumpTo,mutableState.completed,mutableState.failed)
            self.listObjects = (tuple(mutableState.heap),
                                tuple(mutableState.registers),
                                tuple(mutableState.varNameList),
                                tuple(mutableState.calls))

        heap = property(lambda self: list(self.listObjects[0]))
        registers = property(lambda self: list(self.listObjects[1]))
        varNameList = property(lambda self: list(self.listObjects[2]))
        calls = property(lambda self: list(self.listObjects[3]))
        pc = property(lambda self: self.flagObjects[0])
        jumpTo = property(lambda self: self.flagObjects[1])
        completed = property(lambda self: self.flagObjects[2])
        failed = property(lambda self: self.flagObjects[3])

        # the default hash for tuples-of-tuples in python will not
        # work, so you need to go all the way down a tuple tree
        # explicitly to correctly distinguish things

        def __hash__(self): 
            h =  self._tupleHash(self.flagObjects) ^ self._tupleHash(self.listObjects)
            return h

        def __eq__(self,other): 
            return self._tupleEq(self.flagObjects,other.flagObjects) \
                   and self._tupleEq(self.listObjects,other.listObjects)
        
        # walk down a tree of tuples and find a hash for the tree
        def _tupleHash(self,x):
            if (isinstance(x,tuple)):
                h = 0
                for i in xrange(len(x)):
                    h = h ^ self._tupleHash(x[i])
                return h
            else:
                return hash(x)                

        # walk down a two trees of tuples and compare them
        def _tupleEq(self,x,y):
            if (isinstance(x,tuple) and isinstance(y,tuple) and len(x)==len(y)):
                for i in xrange(len(x)):
                    if not self._tupleEq(x[i],y[i]): return False
                return True
            else:
                return x==y                

        def mutableVersion(self):
            """Restore from a copy produced by save()."""
            result = Interpreter.MutableState()
            result.heap = self.heap
            result.registers = self.registers
            result.varNameList = self.varNameList
            result.calls = list(self.calls)
            (result.pc,result.jumpTo,result.completed,result.failed) = self.flagObjects
            return result

    #
    # Interpreter methods start here
    # 

    def __init__(self,wamprog,plugins=[],debugMode=DEFAULT_DEBUG_MODE):
        self.debugMode = debugMode
        self.state = Interpreter.MutableState()
        self.wamprog = wamprog
        self.constantTable = symtab.SymbolTable()
        self.plugins = plugins + wamplugin.builtInPlugins()
        # to answer a query, we add to the end of the compiled
        # program. this records the end of the compiled program,
        # so we can clear out previously-compiled queries
        self.coreProgramSize = 0

    def clear(self):
        self.state = Interpreter.MutableState()

    def addPlugins(self,plugins):
        self.plugins += plugins
        
    def saveState(self):
        return self.state.immutableVersion()

    def restoreState(self,immutableState):
        self.state = immutableState.mutableVersion()

    def executeWithoutBranching(self,startAddr=-1):
        """Execute instructions until a conditional opcode fails, the
        top-level program return, or a 'callp' instruction is
        reached. Returns an empty dictionary.  All other status is
        encoded in the wam's state: completed, failed, and/or jumpTo.

        """

        if startAddr>=0: self.state.pc = startAddr
        self.state.failed = False
        self.state.completed = False

        #execute opcodes until we fail, the program completes, or we
        #hit a 'callp' opcode - indicated by setting pc=-1 

        while not self.state.failed and not self.state.completed and self.state.pc>=0:
            inst = self.wamprog.instructions[self.state.pc]
            opcode = inst[0]
            args = inst[1:]
            self.nextCall = None
            # Use python's reflection capabilities to dispatch to the
            # appropriate opcode method.  Each opcode should be
            # implemented as a method of Interpreter.
            opcodeMethod = self.__class__.__dict__.get(opcode)
            assert opcodeMethod, opcode+' not implemented'
            opcodeMethod(self,*args)
            if self.debugMode: logging.debug('at %d inst %s: state %s' % (self.state.pc,str(inst),str(self.state)))

        return

    #
    # Implementation of opcodes. The convention here is that both
    # registers and heap positions are references relative to the TOP
    # of the register/heap stack, eg a variable index 'a' and
    # 'relativeHeapIndex' should be small negative numbers, -1, -2...
    #

    def allocate(self,n,varNames=None):
        """Allocate n new variable registers, associated with the
        given variable names.  These will be accessed as
        self.registers[self.rp + a], and will hold indices into
        the heap."""
        assert n>=0
        self.state.registers += [None]*n
        if self.debugMode:
            #create meaningful names for the newly allocated variables
            if not varNames: varNames="ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
            assert len(varNames)==n
            cp = len(self.state.calls)
            if cp>0:
                depthAnnotatedVarNames = map(lambda s:s+'_'+str(cp), varNames)
                self.state.varNameList += depthAnnotatedVarNames
            else:
                self.state.varNameList += varNames
        self.state.pc += 1

    def callp(self,pred):
        """Insert an appropriate CallStackFrame on the call stack for a later
        'returnp' call, and then mark the interpreter's as ready to
        jump to that predicate's definition."""
        self.state.pc += 1   #return to following instruction
        self.state.calls.append(Interpreter.CallStackFrame(self.state))
        self.state.jumpTo = pred
        self.state.pc = -1

    def returnp(self):
        """Mark as completed if we're at the top level, and otherwise, pop a
        CallStackFrame and return to that state. """
        if not self.state.calls:
            self.state.completed = True
        else:
            frame = self.state.calls.pop()
            if self.debugMode:
                #sanity check
                for i in range(frame.hp):
                    if self.state.hasVarAtI(i) and self.state.getVarValueAtI(i)>=frame.hp:
                        print 'problem return! frame',frame,'after pop',self.state
                        assert False
            self.state.heap = self.state.heap[:frame.hp]
            self.state.registers = self.state.registers[:frame.rp]
            self.state.varNameList = self.state.varNameList[:frame.rp]
            self.state.pc = frame.pc
            self.state.jumpTo = frame.jumpTo

    #
    # pushing values onto the (top of the) heap
    #

    def pushconst(self,a):
        """Add a constant a to the heap."""
        id = self.constantTable.getId(a)
        self.state.heap.append(self.state.createConstCellById(id))
        self.state.pc += 1

    def pushfreevar(self,a):
        """Add an unbound variable to the heap, and have the a-th
        local variable point to it."""
        assert a<0 
        i = len(self.state.heap)
        self.state.heap.append(self.state.createVarCell(i))
        self.state.registers[a] = i
        self.state.pc += 1

    def pushboundvar(self,a):
        """Push the value of the a-th local variable onto the heap."""
        assert a<0
        valueOfA = self.state.deref(self.state.registers[a])
        self.state.heap.append(self.state.createVarCell(valueOfA))
        self.state.pc += 1

    #
    # Matching register variables against the heap.  
    #

    def unifyconst(self,a,relativeHeapIndex):
        """Check that constant a is equal to something stored in the heap.  If
        the heap cell is a free variable, then bind it to the
        constant.  If the heap cell is bound to different constant,
        fail."""
        assert relativeHeapIndex<0
        i = len(self.state.heap)+relativeHeapIndex
        ri = self.state.deref(i)
        if self.state.hasConstAtI(ri):
            if not self.constantTable.hasId(a): #this constant has not been stored anywhere in the heap
                self.state.failed = True
            else:
                aid  = self.constantTable.getId(a)
                self.state.failed = (self.state.getIdOfConstValueAtI(ri)!=aid)
        else:
            assert self.state.hasFreeVarAtI(ri)
            self.state.heap[ri] = self.state.createConstCellById(self.constantTable.getId(a))
            self.state.collapsePointers(i,ri)
        self.state.pc += 1

    def initfreevar(self,a,relativeHeapIndex):
        """Bind a free variable to this heap position"""
        assert a<0 and relativeHeapIndex<0
        self.state.registers[a] = len(self.state.heap)+relativeHeapIndex
        self.state.pc += 1
    
    def unifyboundvar(self,a,relativeHeapIndex):
        """Unify a variable to a heap position.
        """
        assert a<0 and relativeHeapIndex<0
        state = self.state
        #convert to absolute heap indices
        i = len(state.heap) + relativeHeapIndex
        j = state.registers[a]
        #follow pointer chains
        ri = state.deref(i)
        rj = state.deref(j)
        #cases for unification
        if ri==rj: 
            pass #ok
        elif state.hasConstAtI(ri) and state.hasConstAtI(rj):
            state.failed = (state.getIdOfConstValueAtI(ri) != state.getIdOfConstValueAtI(rj))
        elif state.hasConstAtI(ri):
            assert state.hasFreeVarAtI(rj)
            rj = ri
        elif state.hasConstAtI(rj): 
            assert state.hasFreeVarAtI(ri)
            state.heap[ri] = state.copyConstCell(rj)
        elif rj>ri:  
            assert state.hasFreeVarAtI(ri) and state.hasFreeVarAtI(rj)
            state.heap[rj] = state.createVarCell(ri)  #bind larger to small
            rj = ri
        else:
            assert ri>rj and state.hasFreeVarAtI(ri) and state.hasFreeVarAtI(rj)
            state.heap[ri] = state.createVarCell(rj)  #bind larger to smaller
            ri = rj
        state.collapsePointers(i,ri)
        state.collapsePointers(j,rj)
        #increment program counter
        state.pc += 1

    #
    # higher-level, no-opcode methods start here
    #

    #
    # used mainly in plugins
    # 

    def getConstantArg(self,k,i):
        """Special accessor to the current state: get the value associated
        with the i-th argument, starting at 1, of a arity-k predicate.
        Return None if it is an unbound variable."""
        rj = self._derefArg(k,i)
        if self.state.hasFreeVarAtI(rj): 
            return None
        else:
            return self._heapIndexToConst(rj)

    def getArg(self,k,i):
        """Special accessor to the current state: get the value associated
        with the i-th argument, starting at 1, of a arity-k predicate.
        Return None if it is an unbound variable."""
        rj = self._derefArg(k,i)
        if self.state.hasFreeVarAtI(rj): 
            return self._heapIndexToVar(rj)
        else:
            return self._heapIndexToConst(rj)

    def setArg(self,k,i,value):
        """Special accessor to the current state: set the value associated
        with the i-th argument, starting at 1, of a arity-k predicate."""
        state = self.state
        j = len(self.state.heap) - k + i - 1
        rj = self.state.deref(j)
        assert state.hasFreeVarAtI(rj) 
        id = self.constantTable.getId(value)
        state.heap[rj] = state.createConstCellById(id)
        state.collapsePointers(j,rj)

    def _derefArg(self,k,i):
        """Heap index for i-th argument of an arity-k predicate."""
        j = len(self.state.heap) - k + i - 1
        return self.state.deref(j)

    def _heapIndexToConst(self,rj):
        assert self.state.hasConstAtI(rj)
        id = self.state.getIdOfConstValueAtI(rj)
        return self.constantTable.getSymbol(id)

    def _heapIndexToVar(self,rj):
        assert self.state.hasFreeVarAtI(rj)
        state = self.state
        for i,varName in enumerate(state.varNameList):
            if state.registers[i]:
                j = state.deref(state.registers[i])
                if j==rj: return varName
        return "V%d" % rj

    #
    # human-readable views of a state
    # 

    def asDict(self,otherState):
        """Convert to a dictionary mapping variable names to values."""
        result = {}
        constants = self.constantTable.getSymbolList()
        for i,varName in enumerate(otherState.varNameList):
            j = otherState.deref(otherState.registers[i])
            if otherState.hasConstAtI(j): 
                constTabIndex = otherState.getIdOfConstValueAtI(j) - 1
                result[varName] = constants[constTabIndex]
            else:
                result[varName] = "X%s" % j
        return result

    def asString(self,otherState):
        """Convert to string description, for debugging purposes."""
        result = ""
        jumps = []
        if otherState.jumpTo: jumps.append(otherState.jumpTo)
        for c in otherState.calls:
            if c.jumpTo: jumps.append(c.jumpTo)
        if jumps: result += "...".join(reversed(jumps))
        result += " {"+",".join(map(lambda (k,v):'%s=%s' %(k,v), self.asDict(otherState).items()))+"}"
        if otherState.completed: result += "*"
        return result

    def canonicalForm(self,rootState,otherState):
        # buffer to hold the canonical version of the state
        form = []
        #first get binding information for vars in the root state
        constants = self.constantTable.getSymbolList()
        for i,varName in enumerate(rootState.varNameList):
            #use fact that varNames and registers are always appended to...
            j = otherState.deref(otherState.registers[i])
            if otherState.hasConstAtI(j): 
                constTabIndex = otherState.getIdOfConstValueAtI(j) - 1
                value = constants[constTabIndex]
            else:
                value = "X%s" % j
            form.append('%s=%s' % (varName,value))
        # next get pending goal information
        saved = self.saveState()
        self.restoreState(otherState)
        while not self.state.completed:
            self.executeWithoutBranching()
            state = self.state
            if self.state.jumpTo:
                #call information
                functor,arityStr = state.jumpTo.split("/")
                arity = int(arityStr)
                form.append(state.jumpTo)
                for i in range(1,arity+1):
                    form.append(self.getArg(arity,i))
                self.returnp()
        self.restoreState(saved)
        return tuple(form)

    def pendingGoals(self,otherState,pprint=True):
        """Convert to a list of pending goals to be proved - if pprint is
        true, in a human-readable format."""
        pending = []
        #backup the current state
        saved = self.saveState()
        self.restoreState(otherState)
        #simulate executing the remainder of the program, till
        #completion, but when there is a 'callp', just emit the
        #current goal and return
        while not self.state.completed:
            self.executeWithoutBranching()
            if self.state.jumpTo:
                pending.append(self._nextPendingGoal(pprint=pprint))
                self.returnp()
        self.restoreState(saved)
        if pprint: return ",".join(pending) 
        else: return tuple(pending)

    def _nextPendingGoal(self,pprint=True):
        """Given that we just did a callp, return a representation of
        #the associated goal"""
        state = self.state
        functor,arityStr = state.jumpTo.split("/")
        arity = int(arityStr)
        result = [functor]
        for i in range(1,arity+1):
            result.append(self.getArg(arity,i))
        if not pprint: return tuple(result)
        elif arity==1: return result[0]
        else: return '%s(%s)' % (result[0],",".join(result[1:]))
            
    def __str__(self,details=False):
        return "wi:"+str(self.state)

    #
    # recommended interface for using an interpreter
    # 

    @staticmethod
    def preloadedInterpreter(programFiles):
        """Given programFiles, a list of file names with extensions of .ppr,
        .graph, .facts, compile everything and create an
        interpreter."""
        plugins = []
        wp = wamcompiler.Program()
        rules = None
        for a in programFiles:
            if a.endswith(".ppr"): 
                print 'compiling proppr file',a
                rules = parser.Parser.parseFile(a,rules)
                wp = wamcompiler.Compiler().compileRules(rules,wp)
            elif a.endswith(".graph"):
                print 'loading graph file',a
                plugins.append(wamplugin.GraphPlugin.load(a))            
            elif a.endswith(".cfacts"): 
                print 'loading fact file',a
                plugins.append(wamplugin.FactPlugin.load(a))
            else:
                assert False,"no idea how to preload "+a
        return Interpreter(wp,plugins)

    def initialState(self,queryString=None,queryRule=None,queryGoals=None):
        """After creating a preloadedInterpreter, create the initial state for the proof graph
        associated with a query."""

        #create the queryRule from the input
        if queryString:
            assert (not queryRule) and (not queryGoals), "Only one of queryString, queryRule, queryGoals should be specified"
            queryRule = parser.Parser.parseQuery(queryString)
        if queryGoals:
            assert (not queryRule) and (not queryString), "Only one of queryString, queryRule, queryGoals should be specified"            
            queryRule = parser.Rule(None,queryGoals)
        assert not queryRule.lhs
        queryRule.variabilize()

        #compile the queryRule into the program
        if not self.coreProgramSize:
            # initialize size of the 'core' program, vs the query program
            self.coreProgramSize = len(self.wamprog.instructions)
        # reset the interpreter, and discard any compiled code added by previous queries
        self.clear()
        self.wamprog.truncateTo(self.coreProgramSize)
        # compile the query onto the end of the program
        wamcompiler.Compiler().compileRule(queryRule,wamprog=self.wamprog)

        # execute query code to get start state
        self.executeWithoutBranching(startAddr=self.coreProgramSize)
        assert not self.state.failed
        return self.saveState()

    def childStates(self,state):
        if state.completed:
            return []
        else:
            for plugin in self.plugins:
                if plugin.claim(state.jumpTo):
                    self.restoreState(state)
                    result = []
                    for child in plugin.outlinks(state,self):
                        result.append(child)
                    return result

            result = []
            assert state.jumpTo in self.wamprog.labels, "Unknown predicate "+state.jumpTo
            for addr in self.wamprog.labels[state.jumpTo]:
                self.restoreState(state)
                self.executeWithoutBranching(startAddr=addr)
                result.append(self.saveState())
            return result

    def solutions(self,**kw):
        """Prove a goal and return all solution states, as dictionary."""
        return [self.asDict(s) for s in self.prove(**kw) if s.completed]

    def prove(self,queryString=None,queryRule=None,queryGoals=None,maxDepth=10):
        """Prove a goal and return all the reachable states."""
        start = self.initialState(queryString=queryString,queryRule=queryRule,queryGoals=queryGoals)
        result = list(self._proveDfs(start,0,maxDepth))
        return result
    
    def _proveDfs(self,state,depth,maxDepth):
        yield state
        if not state.completed and depth<maxDepth:
            for child in self.childStates(state):
                for result in self._proveDfs(child,depth+1,maxDepth):
                    yield result
        

    def trace(self,queryString=None,queryRule=None,queryGoals=None,maxDepth=10):
        """Prove a goal and return all the reachable states."""
        start = self.initialState(queryString=queryString,queryRule=queryRule,queryGoals=queryGoals)
        result = list(self._tracingProveDfs(start,0,maxDepth))
        return result
    
    def _tracingProveDfs(self,state,depth,maxDepth):
        tabbing = '| ' * depth
        if state.completed: 
            print '%s%s' % (tabbing,str(self.asDict(state)))
        else:
            pending = self.pendingGoals(state)
            print '%s%s' % (tabbing,pending)
        yield state
        if not state.completed and depth<maxDepth:
            for child in self.childStates(state):
                for result in self._tracingProveDfs(child,depth+1,maxDepth):
                    yield result

##############################################################################
# test driver
##############################################################################

if __name__ == "__main__":

    wamInterp = Interpreter.preloadedInterpreter(sys.argv[1:-1])
    q = sys.argv[-1]
    print 'tracing',q
    wamInterp.trace(queryString=q)
    print 'answering',q
    for s in wamInterp.solutions(queryString=q):
        print s

