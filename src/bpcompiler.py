# (C) William W. Cohen and Carnegie Mellon University, 2016

# compile a single definite clause into a series of operations that
# perform belief propogation

import sys
import collections
import logging

import ops
import funs
import config
import symtab 
import parser
import declare
import matrixdb
import tensorlog

conf = config.Config()
conf.strict = True;         conf.help.strict =        "Check that a clause fits all assumptions"
conf.trace = False;         conf.help.trace =         "Print debug info during BP"
conf.produce_ops = True;    conf.help.produce_ops =   "Turn off to debug analysis"

#
# helper classes - info on variables and goals
#

class VarInfo(object):
    """Node in a factor graph corresponding to a variable."""
    def __init__(self,v):
        self.var = v               #var name
        self.outputOf = None       #goal index it's output of: -1 for lhs
        self.inputTo = set()       #goal indices this is an input for
        self.connected = False     #set after a signal has been sent, so we can see if graph is connected
    def __repr__(self):
        return 'VarInfo(var=%r,outputOf=%r,inputTo=%r,connected=%r)' % (self.var,self.outputOf,self.inputTo,self.connected)
    
class GoalInfo(object):
    """Node in a factor graph corresponding to a goal with index j in self.goals."""
    def __init__(self,j):
        self.index = j
        self.inputs = set()         #variables that are inputs for this goal
        self.outputs = set()        #variables that are outputs
        self.definedPred = False    #goal is defined by rules, not the matrixdb
    def __str__(self):
        def _ss(s): return "<" + ",".join(map(str,s)) + ">"
        return "\t".join(["+"+_ss(self.inputs), "-"+_ss(self.outputs)])
    def __repr__(self):
        return 'GoalInfo(index=%d,inputs=%r,outputs=%r,defined=%r)' % (self.index,self.inputs,self.outputs,self.definedPred)


#
# main class
# 

class BPCompiler(object):
    """Compiles a logical rule + a mode into a sequence of ops.py operations."""

    def __init__(self,lhsMode,tensorlogProg,depth,rule):
        """ Build a compiler for a rule.  The tensorlogProg is used to
        recursively compile any intensionally-defined predicates.
        The depth is a depth bound.
        """
        self.rule = rule                    #rule to be compiled
        self.lhsMode = lhsMode              #mode with respect to which this will be compiled
        self.tensorlogProg = tensorlogProg  #need back pointer to this to compile subpredicates
        self.depth = depth #used for recursively compiling subpredicates with tensorlogProg
        self.ops = []      #generated list of operations used for BP
        self.output = None #final outputs of the function associated with performing BP for the mode
        self.inputs = None #inputs of the function associated with performing BP for the mode
        self.goals = [self.rule.lhs] + self.rule.rhs  #so we can systematically index goals with an int j
        self.compiled = False               #set to True when compilation is finihe
        if conf.strict: self.validateRuleBeforeAnalysis()

    #
    # compile and then access the result of compilation
    # 

    def getFunction(self): 
        """ After compilation, return a function that performs BP
        """
        if not self.compiled: 
            self.compile()
        return funs.OpSeqFunction(self.inputs, self.output, self.ops, self.rule)

    #
    # debugging tools
    #

    def showVars(self):
        print "\t".join("var outOf inputTo outOf inputTo fb".split())
        for v in sorted(self.varDict.keys()):
            vin = self.varDict[v]
            def _gs(j): 
                if j==None: return 'None'
                else: return str(self.goals[j])
            print "\t".join([ v, str(vin.outputOf), ",".join(map(str,vin.inputTo)), 
                              _gs(vin.outputOf), ",".join(map(_gs,vin.inputTo)),
                              str(vin.connected)])

    def showRule(self):
        #print "\t".join("id goal ins outs roots".split())
        def goalStr(j): return str(self.goalDict[j])
        for j in range(len(self.goals)):
            print '%2d' % j,'\t ',goalStr(j),'\t',str(self.goals[j]),str(self.toMode(j))

    def showOps(self):
        print 'inputs:',",".join(self.inputs)
        print 'output:',",".join(self.output)
        print 'compiled to:'
        for op in self.ops:
            print '\t',op


    def compile(self):
        """Top-level analysis routine for a rule.
        """
        #infer the information flow for all the variables and goals,
        #and store in the varDict/goalDict under vin.outputOf,
        #vin.inputTo, gin.outputs, gin.inputs

        self.inferFlow()
        
        #recursively call the tensorlog program to compile 
        #any intensionally-defined subpredicates
        self.compileDefinedPredicates()

        # generate an operation sequence that implements the BP algorithm
        if conf.produce_ops:
            self.generateOps()

        self.compiled = True

    #
    # simpler subroutines of compile
    #

    def validateRuleBeforeAnalysis(self):
        """Raises error if the rule doesn't satisfy the assumptions made by
        the compiler.  Can be before flow analysis."""
        assert self.rule.lhs.arity==2
        for goal in self.rule.rhs:
            assert goal.arity==1 or goal.arity==2


    def inferFlow(self):
        """ Infer flow of information in the clause, by populating a VarInfo
        object for each variable and a GoalInfo object for each goal.
        Information flows from the lhs's input variable, to the output
        variable through predicates which map inputs to outputs.
        """

        # populate the varDict and goalDict structures for a rule
        self.varDict = {}
        self.goalDict = {}

        #for lhs, infer inputs/outputs from the known mode
        gin = self.goalDict[0] = GoalInfo(0)
        gin.mode = self.lhsMode
        for i in range(self.rule.lhs.arity):
            v = self.rule.lhs.args[i]
            if v not in self.varDict: 
                self.varDict[v] = VarInfo(v)
            else:
                assert False,'same variable cannot appear twice in a rule lhs'
            vin = self.varDict[v]
            assert parser.isVariableAtom(v), 'arguments to defined predicate %s cannot be a constant' % str(self.rule.lhs)
            if gin.mode.isInput(i):
                gin.inputs.add(v) #input to predicate means output of lhs
                vin.outputOf = 0
            else:
                gin.outputs.add(v)  #input to predicate means input to lhs
                vin.inputTo.add(0)

        # for rhs goals, use inputs/outputs to infer mode
        for j in range(1,len(self.goals)):
            gin = self.goalDict[j] = GoalInfo(j)
            goal = self.goals[j]
            for i in range(goal.arity):
                v = goal.args[i]
                if parser.isVariableAtom(v):
                    if v not in self.varDict: self.varDict[v] = VarInfo(v)
                    vin = self.varDict[v]
                    if vin.outputOf!=None:
                        # not first occurrence, so it's an input to this goal
                        gin.inputs.add(v)
                        vin.inputTo.add(j)
                    else:
                        gin.outputs.add(v)
                        vin.outputOf = j

        #validate - lhs has exactly one output, which must be bound somewhere 
        lhsGin = self.goalDict[0]
        assert len(lhsGin.outputs)==1, 'lhs mode '+str(self.lhsMode)+' and lhs has >1 output: outputs '+str(lhsGin.outputs)+' for rule '+str(self.rule)
        y = only(lhsGin.outputs)
        self.varDict[y]!=None,'lhs output variable "%s" not bound' % y

    def compileDefinedPredicates(self):
        """Recursively call the tensorlog program to compile
        each subpredicate."""
        for j in range(1,len(self.goals)):
            gin = self.goalDict[j]
            mode = self.toMode(j)        
            if self.tensorlogProg.findPredDef(mode):
                gin.definedPred = True
                self.tensorlogProg.compile(mode,self.depth+1)

    def toMode(self,j):
        """Helper - Return a mode declaration for the j-th goal of the rule,
        based on the information flow"""
        goal = self.goals[j]
        gin = self.goalDict[j]
        def argIOMode(x): 
            if x in gin.inputs: return 'i'
            elif x in gin.outputs: return 'o'
            else:
                assert x!='i' and x!='o' and x!='i1' and x!='i2', 'Illegal to use constants i,o,i1,o1 in a program'
                return x
        return declare.ModeDeclaration(parser.Goal(goal.functor, [argIOMode(x) for x in goal.args]), strict=False)

    #
    # the main belief propagation algorithm
    #

    def generateOps(self):
        """Emulate BP and emit the sequence of operations needed.  Instead of
        actually constructing a message from src->dst in the course of
        BP, what we do instead is emit operations that would construct
        the message and assign it a 'variable' named 'foo', and then
        return not the message but the variable-name string 'foo'. """

        messages = {}

        def addOp(op,traceDepth,msgFrom,msgTo):
            """Add an operation to self.ops, echo if required"""
            if conf.trace: print '%s+%s' % (('| '*traceDepth),op)
            def jToGoal(msg): return str(self.goals[msg]) if type(msg)==type(0) else msg
            op.setMessage(jToGoal(msgFrom),jToGoal(msgTo))
            self.ops.append(op)

        def msgGoal2Var(j,v,traceDepth):
            """Send a message from a goal to a variable.  Note goals can have at
            most one input and at most one output.  This is complex
            because there are several cases, depending on if the goal
            is LHS on RHS, and if the variable is an input or
            output."""
            gin = self.goalDict[j]
            if conf.trace: print '%smsg: %d->%s' % (('| '*traceDepth),j,v)
            # The lhs goal, j==0, is the input factor
            if j==0 and v in self.goalDict[j].inputs:
                #input port -> input variable
                assert parser.isVariableAtom(v),'input must be a variable'
                return v
            elif j==0:
                #output port -> output variable
                assert False,'illegal message - something is wrong'
            elif j>0 and v in self.goalDict[j].outputs:
                #message from rhs goal to an output variable of that goal
                msgName = 'f_%d_%s' % (j,v) 
                mode = self.toMode(j)
                if not gin.inputs:
                    # special case - binding a variable to a constant with set(Var,const)
                    assert matrixdb.isAssignMode(mode),'output variables without inputs are only allowed for assign/2'
                    addOp(ops.AssignOnehotToVar(msgName,mode), traceDepth,j,v)
                    return msgName
                else:
                    fx = msgVar2Goal(only(gin.inputs),j,traceDepth+1) #ask for the message forward from the input to goal j
                    if ops.isBuiltinIOOp(mode):
                        addOp(ops.BuiltInIOOp(msgName,fx,mode), traceDepth,j,v)
                    elif not gin.definedPred:
                        addOp(ops.VecMatMulOp(msgName,fx,mode), traceDepth,j,v)
                    else:
                        addOp(ops.DefinedPredOp(self.tensorlogProg,msgName,fx,mode,self.depth+1), traceDepth,j,v)
                    return msgName
            elif j>0 and v in self.goalDict[j].inputs:
                #message from rhs goal to an input variable of that goal
                gin = self.goalDict[j]
                msgName = 'b_%d_%s' % (j,v) 
                mode = self.toMode(j)
                def hasOutputVarUsedElsewhere(gin): 
                    outVar = only(gin.outputs)
                    return self.varDict[outVar].inputTo
                if gin.outputs and hasOutputVarUsedElsewhere(gin):
                    bx = msgVar2Goal(only(gin.outputs),j,traceDepth+1) #ask for the message backward from the input to goal 
                    addOp(ops.VecMatMulOp(msgName,bx,mode,transpose=True), traceDepth,j,v)
                    return msgName
                else:
                    if gin.outputs:
                        #optimize away the message from the output
                        # var of gin, since it would be a dense
                        # all-ones vector
                        assert len(gin.outputs)==1, 'need single output from %s' % self.goals[j]
                        #this variable now is connected to the main chain
                        self.varDict[only(gin.outputs)].connected = True
                        assert not ops.isBuiltinIOOp(mode), 'can only use built in io operators where inputs and outputs are used'
                        assert not gin.definedPred, 'subpredicates must generate an output which is used downstream'
                        addOp(ops.AssignPreimageToVar(msgName,mode), traceDepth,j,v)
                    else:
                        addOp(ops.AssignVectorToVar(msgName,mode), traceDepth,j,v)
                        
                    return msgName
            else:
                assert False,'unexpected message goal %d -> %s ins %r outs %r' % (j,v,gin.inputs,gin.outputs)

        def msgVar2Goal(v,j,traceDepth):
            """Message from a variable to a goal.
            """
            vin = self.varDict[v]
            vin.connected = True
            gin = self.goalDict[j]
            #variables have one outputOf, but possily many inputTo connections
            vNeighbors = [j2 for j2 in [vin.outputOf]+list(vin.inputTo) if j2!=j]
            if conf.trace: print '%smsg from %s to %d, vNeighbors=%r' % ('| '*traceDepth,v,j,vNeighbors)
            assert len(vNeighbors),'variables should have >1 neighbor but %s has only one: %d' % (v,j)
            #form product of the incoming messages, cleverly
            #generating only the variables we really need
            currentProduct = msgGoal2Var(vNeighbors[0],v,traceDepth+1)
            for j2 in vNeighbors[1:]:
                nextProd = 'p_%d_%s_%d' % (j,v,j2) if j2!=vNeighbors[-1] else 'fb_%s' % v
                multiplicand = msgGoal2Var(j2,v,traceDepth+1)
                addOp(ops.ComponentwiseVecMulOp(nextProd,currentProduct,multiplicand), traceDepth,v,j)
                currentProduct = nextProd
            return currentProduct

        #
        # main BP code starts here
        #

        #generate a message from the output variable to the lhs
        outputVar = only(self.goalDict[0].outputs)
        outputMsg = msgVar2Goal(outputVar,0,1)

        #now look for other unconnected variables, and connect them to
        #a pseudo-node so they have something to send to.  The
        #outputMsg above will be weighted by the product of all of
        #these messages.
        weighters = []
        psj = len(self.goals)
        self.goals.append( parser.Goal('PSEUDO',[]) )
        self.goalDict[psj] = GoalInfo(psj)
        #heuristic - start with the rightmost unconnected variable,
        #hoping that it's the end of a chain rooted at the input,
        #which should be quicker to evaluate
        for j in reversed(range(1,len(self.goals))):
            goalj = self.goals[j]
            for i in range(goalj.arity):
                v = goalj.args[i]
                if parser.isVariableAtom(v) and not self.varDict[v].connected:
                    #save the message from this unconnected node
                    weighters.append(msgVar2Goal(v,psj,1)) 
        #multiply the weighting factors from the unconnected node to
        #the outputMsg, again cleverly reusing variable names to keep
        #the expression simple.
        currentProduct = outputMsg
        for msg in weighters:
            nextProd = 'w_%s' % outputVar if msg==weighters[-1] else 'p_%s_%s' % (msg,outputVar)
            multiplicand = msg
            addOp(ops.WeightedVec(nextProd,multiplicand,currentProduct),0,msg,'PSEUDO')
            currentProduct = nextProd

        # save the output and inputs 
        self.output = currentProduct
        self.inputs = list(self.goalDict[0].inputs)

def only(c):
    """Return only member of a singleton set, or raise an error if the set's not a singleton."""
    assert len(c)==1,'non-singleton ' + repr(c)
    for elt in c: return elt

#
# a test driver
#

if __name__ == "__main__":
    conf.pprint()

    if len(sys.argv)<2:
        print 'usage: rule mode'
        sys.exit(-1)

    conf.strict = False
    #conf.produce_ops = False
    p = parser.Parser()

    ruleString = sys.argv[1]
    rule = p.parseRule(ruleString)
    mode = declare.ModeDeclaration(sys.argv[2])
    rules = parser.RuleCollection()
    rules.add(rule)
    prog = tensorlog.Program(db=None,rules=rules)

    c = BPCompiler(mode,prog,0,rule)
    c.compile()
    c.showRule()
    c.showVars()
    c.showOps()



