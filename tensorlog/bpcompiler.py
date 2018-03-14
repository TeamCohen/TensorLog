# (C) William W. Cohen and Carnegie Mellon University, 2016

# compile a single definite clause into a series of operations that
# perform belief propogation

import sys
import collections
import logging

from tensorlog import ops
from tensorlog import funs
from tensorlog import config
from tensorlog import parser
from tensorlog import declare
from tensorlog import matrixdb

conf = config.Config()
conf.strict = True;     conf.help.strict =    "Check that a clause fits all assumptions"
conf.trace = False;     conf.help.trace =     "Print debug info during BP"
conf.produce_ops = True;  conf.help.produce_ops =   "Turn off to debug analysis"

# functor for the special 'assign(Var,const) predicate
ASSIGN = 'assign'

#
# helper classes - info on variables and goals
#

class VarInfo(object):
  """Node in a factor graph corresponding to a variable."""
  def __init__(self,v):
    self.var = v             #var name
    self.outputOf = None     #goal index it's output of: -1 for lhs
    self.inputTo = set()     #goal indices this is an input for
    self.connected = False   #set after a signal has been sent, so we can see if graph is connected
    self.varType = None      #type of variable, according to type declarations in the matrixDB
  def __repr__(self):
    return 'VarInfo(var=%r,outputOf=%r,inputTo=%r,connected=%r)' % (self.var,self.outputOf,self.inputTo,self.connected)

class GoalInfo(object):
  """Node in a factor graph corresponding to a goal with index j in self.goals."""
  def __init__(self,j):
    self.index = j
    self.inputs = set()       #variables that are inputs for this goal
    self.outputs = set()      #variables that are outputs
    self.definedPred = False  #goal is defined by rules, not the matrixdb
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
    #mode with respect to which this will be compiled
    self.lhsMode = lhsMode
    #back pointer, used to compile subpredicates, and to access db for declarations
    self.tensorlogProg = tensorlogProg
    #used for recursively compiling subpredicates with tensorlogProg
    self.depth = depth
    #rule to be compiled
    self.rule = self.reorderRHS(rule,lhsMode)

    #generated list of operations used for BP
    self.ops = []
    # types of variables generated for the ops
    self.msgType = {}
    #final outputs of the function associated with performing BP for the mode
    self.output = None
    self.outputType = None
    self.inputTypes = None
    #inputs of the function associated with performing BP for the mode
    self.inputs = None
    #so we can systematically index goals with an int j, 0<=j<=n
    self.goals = [self.rule.lhs] + self.rule.rhs
    #set to True when compilation is finished
    self.compiled = False

    if conf.strict: self.validateRuleBeforeAnalysis()

  #
  # compile and then access the result of compilation
  #

  def getFunction(self):
    """ Return a function that performs BP, invoking compiler if needed
    """
    if not self.compiled:
      self.compile()
    return funs.OpSeqFunction(self.inputs, self.output, self.ops, self.rule, self.inputTypes, self.outputType)

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


  #
  #
  #

  def inferredTypes(self):
    self.inferFlow()
    self.inferTypes()
    return dict((v, self.varDict[v].varType) for v in self.varDict)

  #
  # main compilation routine
  #

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

    # infer types for each variable
    self.inferTypes()

    # generate an operation sequence that implements the BP algorithm
    if conf.produce_ops:
      self.generateOps()

    # forward type information to the ops
    for op in self.ops:
      op.dstType = self.msgType.get(op.dst)

    self.compiled = True

  #
  # simpler subroutines of compile
  #
  def reorderRHS(self,rule,lhsMode):
    """ Return a copy of the rule with the goals in the rhs of the rule re-ordered
    so that execution can be mostly left-to-right
    """
    boundVars = set([rule.lhs.args[i] for i in range(rule.lhs.arity) if lhsMode.isInput(i)])
    reorderedRHS = []
    goalsToInclude = rule.rhs
    def readyToExecute(goal):
      inputVars = [a for a in goal.args if (parser.isVariableAtom(a) and (a in boundVars))]
      return inputVars or (goal.functor==ASSIGN and (2 <= goal.arity <= 3)) or goal.arity==1
    def varsIn(goal):
      return set([a for a in goal.args if parser.isVariableAtom(a)])
    while goalsToInclude:
      postponedGoals = []
      progessMade = False
      for goal in goalsToInclude:
        if not readyToExecute(goal):
          postponedGoals.append(goal)
        else:
          reorderedRHS.append(goal)
          boundVars = boundVars.union(varsIn(goal))
          progessMade = True
      goalsToInclude = postponedGoals
      assert progessMade,'cannot order goals for mode %r and rule %s' % (lhsMode,rule.asString())
    result = parser.Rule(rule.lhs,reorderedRHS,rule.features,rule.findall)
    return result

  def validateRuleBeforeAnalysis(self):
    """Raises error if the rule doesn't satisfy the assumptions made by
    the compiler.  Can be before flow analysis."""
    assert self.rule.lhs.arity==2, "Bad arity in rule lhs"
    for goal in self.rule.rhs:
      if goal.functor==ASSIGN:
        assert goal.arity>=1 and goal.arity<=3, "Bad arity in rhs goal '%s'" % goal
      else:
        #TOFIX: multiple input user-defs would need this relaxation
        #elif not self.tensorlogProg.plugins.isDefined(functor=goal.functor, arity=goal.arity):
        assert goal.arity>=1 and goal.arity<=2, "Bad arity in rhs goal '%s'" % goal

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
    y = _only(lhsGin.outputs)
    self.varDict[y]!=None,'lhs output variable "%s" not bound' % y

  def inferTypes(self):
    """ Infer the type of each variable, by populating varInfo.varType
    """
    def informativeType(t): return (t is not None) and (t != matrixdb.THING)

    for i in range(1,len(self.goals)):
      gin = self.goalDict[i]
      functor = self.goals[i].functor
      arity = self.goals[i].arity
      mode = self.toMode(i)
      if self.tensorlogProg.plugins.isDefined(mode):
        for j in range(arity):
          if mode.isOutput(j):
            vj = self.goals[i].args[j]
            self.varDict[vj].varType = self.tensorlogProg.plugins.outputType(mode,self.collectInputTypes(i))
      elif functor == ASSIGN and arity==3:
        # goal is assign(Var,constantValue,type)
        self.varDict[self.goals[i].args[0]].varType = self.goals[i].args[2]
      elif functor != ASSIGN:
        if self.tensorlogProg.findPredDef(mode):
          pass # we handled this case in compileDefinedPredicates
        else:
          # infer type for a database predicate
          for j in range(arity):
            vj = self.goals[i].args[j]
            newTj = self.tensorlogProg.db.schema.getArgType(functor,arity,j)
            oldTj = self.varDict[vj].varType
            if informativeType(newTj):
              if informativeType(oldTj) and oldTj!=newTj:
                logging.warn('variable %s has types %s and %s in rule %s' % (vj,oldTj,newTj,str(self.rule)))
              self.varDict[vj].varType = newTj

  def compileDefinedPredicates(self):
    """Recursively call the tensorlog program to compile
    each subpredicate."""
    for j in range(1,len(self.goals)):
      gin = self.goalDict[j]
      mode = self.toMode(j)
      if self.tensorlogProg.findPredDef(mode):
        gin.definedPred = True
        fun = self.tensorlogProg.compile(mode,self.depth+1)
        # save type information while we're in here, if it's available
        typeForV = fun.inputTypes
        if typeForV is not None:
          for v,typeForV in zip(gin.inputs,fun.inputTypes):
            self.varDict[v].varType = typeForV
          v = _only(gin.outputs)
          self.varDict[v].varType = fun.outputType

  def toMode(self,j):
    """Helper - Return a mode declaration for the j-th goal of the rule,
    based on the information flow"""
    goal = self.goals[j]
    gin = self.goalDict[j]
    def argIOMode(x):
      if x in gin.inputs: return 'i'
      elif x in gin.outputs: return 'o'
      else:
        # TODO figure out what's happening here?
        assert x!='i' and x!='o' and x!='i1' and x!='i2', 'Illegal to use constants i,o,i1,o1 in a program'
        return x
    return declare.ModeDeclaration(parser.Goal(goal.functor, [argIOMode(x) for x in goal.args]), strict=False)

  def collectInputTypes(self,j):
      """Helper - collect a list of all input types for the j-th goal of the rule."""
      inputTypes = []
      mode = self.toMode(j)
      for i in range(mode.arity):
          if mode.isInput(i):
              vi = self.goals[j].args[i]
              inputTypes.append(self.varDict[vi].varType)
      return inputTypes

  #
  # the main belief propagation algorithm
  #

  def generateOps(self):
    """Emulate BP and emit the sequence of operations needed.  Instead of
    actually constructing a message from src->dst in the course of
    BP, what we do instead is emit operations that would construct
    the message and assign it a 'variable' named 'foo', and then
    return not the message but the string that names the 'variable'.
    """

    messages = {}

    def addOp(op,traceDepth,msgFrom,msgTo):
      """Add an operation to self.ops, echo if required"""
      if conf.trace: print '%s+%s' % (('| '*traceDepth),op)
      def jToGoal(msg): return str(self.goals[msg]) if type(msg)==type(0) else msg
      op.setMessage(jToGoal(msgFrom),jToGoal(msgTo))
      self.ops.append(op)

    def makeMessageName(stem,v,j=None,j2=None):
      """ create a string that meaningfully names this message
      """
      msgName = '%s_%s' % (stem,v)
      if j is not None: msgName += '%d' % j
      if j2 is not None: msgName += '%d' % j2
      self.msgType[msgName] = self.varDict[v].varType
      return msgName

    def msgGoal2Var(j,v,traceDepth):
      """Send a message from a goal to a variable.  Note goals can have at
      most one input and at most one output.  This is complex
      because there are several cases, depending on if the goal
      is LHS on RHS, and if the variable is an input or
      output."""
      gin = self.goalDict[j]
      if conf.trace: print '%smsg: %d->%s' % (('| '*traceDepth),j,v)
      if j==0 and v in self.goalDict[j].inputs:
        #input port -> input variable - The lhs goal, j==0, is the input factor
        assert parser.isVariableAtom(v),'input must be a variable'
        return v
      elif j==0:
        #output port -> output variable
        assert False,'illegal message goal %d to var %s' % (j,v)
      elif j>0 and v in self.goalDict[j].outputs:
        #message from rhs goal to an output variable of that goal
        msgName = makeMessageName('f',v,j)
        mode = self.toMode(j)
        if not gin.inputs:
          # special case - binding a variable to a constant with assign(Var,const) or assign(Var,type,const)
          # TODO: should unary predicates in general be an input?
          errorMsg = 'output variables without inputs are only allowed for assign/2 or assign/3: goal %r rule %r' % (str(self.rule.rhs[gin.index-1]),self.rule.asString())
          assert (mode.functor==ASSIGN and mode.arity>=2 and mode.isOutput(0)), errorMsg
          addOp(ops.AssignOnehotToVar(msgName,mode), traceDepth,j,v)
          return msgName
        else:
          # figure out how to forward message from inputs to outputs
          if (self.tensorlogProg.plugins.isDefined(mode)):
            fxs = []
            for vIn in gin.inputs:
              vMsg = msgVar2Goal(vIn,j,traceDepth+1)
              fxs.append(vMsg)
            outType = self.tensorlogProg.plugins.outputType(mode,self.collectInputTypes(j))
            addOp(ops.CallPlugin(msgName,fxs,mode,dstType=outType), traceDepth,j,v)
          else:
            fx = msgVar2Goal(_only(gin.inputs),j,traceDepth+1) #ask for the message forward from the input to goal j
            if not gin.definedPred:
              addOp(ops.VecMatMulOp(msgName,fx,mode), traceDepth,j,v)
            else:
              addOp(ops.DefinedPredOp(self.tensorlogProg,msgName,fx,mode,self.depth+1), traceDepth,j,v)
          return msgName
      elif j>0 and v in self.goalDict[j].inputs:
        #message from rhs goal to an input variable of that goal
        gin = self.goalDict[j]
        msgName = makeMessageName('b',v,j)
        mode = self.toMode(j)
        def hasOutputVarUsedElsewhere(gin):
          outVar = _only(gin.outputs)
          return self.varDict[outVar].inputTo
        if gin.outputs and hasOutputVarUsedElsewhere(gin):
          bx = msgVar2Goal(_only(gin.outputs),j,traceDepth+1) #ask for the message backward from the input to goal
          addOp(ops.VecMatMulOp(msgName,bx,mode,transpose=True), traceDepth,j,v)
          return msgName
        else:
          if gin.outputs:
            #optimize away the message from the output
            # var of gin, since it would be a dense
            # all-ones vector
            # TODO: is this needed, since multiplication by an all-ones vector is kindof how this is usually implemented?
            assert len(gin.outputs)==1, 'need single output from %s' % self.goals[j]
            #this variable now is connected to the main chain
            self.varDict[_only(gin.outputs)].connected = True
            assert not gin.definedPred, 'subpredicates must generate an output which is used downstream'
            addOp(ops.AssignPreimageToVar(msgName,mode,self.msgType[msgName]), traceDepth,j,v)
          elif self.tensorlogProg.plugins.isDefined(mode):
            addOp(ops.CallPlugin(msgName,[],mode,self.msgType[msgName]), traceDepth,j,v)
          else:
            addOp(ops.AssignVectorToVar(msgName,mode,self.msgType[msgName]), traceDepth,j,v)

          return msgName
      else:
        assert False,'unexpected message goal %d -> %s ins %r outs %r' % (j,v,gin.inputs,gin.outputs)

    def msgVar2Goal(v,j,traceDepth):
      """Message from a variable to a goal.
      """
      vin = self.varDict[v]
      vin.connected = True
      gin = self.goalDict[j]
      #variables have one outputOf, but possily many inputTo
      #connections. Information  propagates back from things the
      #variables are inputTo, unless those goals are CallPlugin's.
      vNeighbors = [j2 for j2 in [vin.outputOf]+list(vin.inputTo) if j2!=j]
      if conf.trace: print '%smsg from %s to %d, vNeighbors=%r' % ('| '*traceDepth,v,j,vNeighbors)
      assert len(vNeighbors),'variables should have >=1 neighbor but %s has none: %d' % (v,j)
      #form product of the incoming messages, cleverly
      #generating only the variables we really need
      currentProduct = msgGoal2Var(vNeighbors[0],v,traceDepth+1)
      for j2 in vNeighbors[1:]:
        nextProd = makeMessageName('p',v,j,j2) if j2!=vNeighbors[-1] else makeMessageName('fb',v)
        multiplicand = msgGoal2Var(j2,v,traceDepth+1)
        addOp(ops.ComponentwiseVecMulOp(nextProd,currentProduct,multiplicand), traceDepth,v,j)
        currentProduct = nextProd
      return currentProduct

    #
    # main BP code starts here
    #

    #generate a message from the output variable to the lhs
    outputVar = _only(self.goalDict[0].outputs)
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
      nextProd = makeMessageName('w',outputVar) if msg==weighters[-1] else makeMessageName('p_%s' % msg,outputVar)
      multiplicand = msg
      addOp(ops.WeightedVec(nextProd,multiplicand,currentProduct),0,msg,'PSEUDO')
      currentProduct = nextProd

    # save the output and inputs
    self.output = currentProduct
    self.outputType = self.msgType[currentProduct]
    self.inputs = list(self.goalDict[0].inputs)
    self.inputTypes = map(lambda v:self.varDict[v].varType, self.inputs)

def _only(c):
  """Return only member of a singleton set, or raise an error if the set's not a singleton."""
  assert len(c)==1,'non-singleton ' + repr(c)
  for elt in c: return elt
