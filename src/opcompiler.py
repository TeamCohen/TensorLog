# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# OBSOLETE VERSION
#

import sys
import ops
import symtab 
import collections
import parser
import tensorlog
import logging

# check that a clause fits assumptions
STRICT = True

def only(c):
    """Return only member of a singleton set, or raise an error if the set's not a singleton."""
    assert len(c)==1,'non-singleton ' + repr(c)
    for elt in c: return elt

#
# helper classes - info on variables and goals
#

class VarInfo(object):
    def __init__(self,v):
        self.var = v               #var name
        self.outputOf = None       #goal index it's output of: -1 for lhs
        self.onehot = False        #
        self.inputTo = set()       #goal indices this is an input for
        self.constraintsDone = False #see detectSecondaryChainConstraints
        self.constrainedVersion = v
    def __repr__(self):
        return 'VarInfo(var=%r,outputOf=%r,onehot=%r,inputTo=%r)' % (self.var,self.outputOf,self.onehot,self.inputTo)
    
class GoalInfo(object):
    def __init__(self,i):
        self.index = i              #index of this goal in the rule's rhs (-1 for the lhs goal)
        self.inputs = set()         #variables that are inputs for this goal
        self.outputs = set()        #variables that are outputs
        self.rootAncestors = set()  #goal indices of all 'root' nodes that feed into this goal
        self.mainChain = False      #is the goal on the 'main chain', i.e. the path connecting the lhs input to the lhs output
        self.definedPred = False    #goal is defined by rules, not a matrix
    def __str__(self):
        def _ss(s): return "<" + ",".join(map(str,s)) + ">"
        return "\t".join(["+"+_ss(self.inputs), "-"+_ss(self.outputs), "anc"+_ss(self.rootAncestors), str(self.mainChain)])
    def __repr__(self):
        return 'GoalInfo(index=%r,inputs=%r,outputs=%r,rootAncestors=%r)' % (self.index,self.inputs,self.outputs,self.rootAncestors)

#
# main class
# 

def buildNullFunction(lhsMode):
    """Build a OpFunction which returns an empty set
    """
    #TODO something wonky about the x/y indices?
    inputs = [('X%d' % i)  for i in range(lhsMode.arity) if lhsMode.isInput(i)]
    outputs = [('Y%d' % i) for i in range(lhsMode.arity) if lhsMode.isOutput(i)]
    assert len(outputs)==1, 'multiple or zero outputs not implemented yet'
    return ops.OpFunction(inputs, outputs, ops.ClearVar(outputs[0]))

class OpCompiler(object):
    """Compiles a logical rule + a mode into a sequence of operations."""

    def __init__(self,tensorlogProg,depth,rule):
        """ Build a compiler for a rule.  The tensorlogProg is used to
        recursively compile any intensionally-defined predicates.
        The depth is a depth bound.
        """
        self.rule = rule
        self.tensorlogProg = tensorlogProg
        self.depth = depth
        self.incoming = []
        self.ops = []
        if STRICT: self.validateRuleBeforeAnalysis()
    
    def validateRuleBeforeAnalysis(self):
        """Raises error if the rule doesn't satisfy the assumptions made by
        the compiler.  Can be run before flow analysis."""
        assert self.rule.lhs.arity==2
        for goal in self.rule.rhs:
            assert goal.arity==1 or goal.arity==2

    def compile(self,lhsMode):
        """Top-level analysis routine for a rule.
        """

        #infer the information flow for all the variables and goals,
        #and store in the varDict/goalDict under vin.outputOf,
        #vin.inputTo, gin.outputs, gin.inputs, and rootAncestors
        self.inferFlow(lhsMode)
        
        #handle incoming chains
        self.removeIncomingChains()
        if STRICT and self.incoming:
            assert False,logging.error('incoming chains are unsupported: ' + ",".join(map(str,self.incoming)))
        #recompute flow for part of the clause w/o incoming chains, and validate
        self.inferFlow(lhsMode)

        #recursively call the tensorlog program to compile 
        #any intensionally-defined subpredicates
        self.compileDefinedPredicates()

        # mark the 'main chain' which links the outputs to the inputs
        self.findMainChain()

        # generate an operation sequence for the main chain 
        self.generateOps()
        return self.ops

    def inferFlow(self,lhsMode):
        """ Infer flow of information in the clause, by populating a VarInfo
        object for each variable and a GoalInfo object for each goal.
        Information flows from the lhs's input variable, to the output
        variable through predicates which map inputs to outputs.
        """
        # (re)populate the varDict and goalDict structures for a rule
        self.varDict = {}
        self.goalDict = {}
        gin = self.goalDict[-1] = GoalInfo(-1)
        gin.mode = lhsMode
        gin.rootAncestors.add(-1)
        #for lhs, infer inputs/outputs from the known mode
        for i in range(self.rule.lhs.arity):
            v = self.rule.lhs.args[i]
            vin = self.varDict[v] = VarInfo(v)
            vin.onehot = lhsMode.isOnehot(i)
            if lhsMode.isInput(i):
                gin.inputs.add(v)
                vin.outputOf = -1
            else:
                vin.inputTo.add(-1)
                gin.outputs.add(v)
        # for rhs goals, use inputs/outputs to infer mode
        for j in range(len(self.rule.rhs)):
            gin = self.goalDict[j] = GoalInfo(j)
            for i in range(self.rule.rhs[j].arity):
                v = self.rule.rhs[j].args[i]
                if parser.isVariableAtom(v):
                    if v not in self.varDict: self.varDict[v] = VarInfo(v)
                    vin = self.varDict[v]
                    if vin.outputOf!=None:
                        gin.inputs.add(v)
                        vin.inputTo.add(j)
                        goalWhichBoundV = vin.outputOf
                        for a in self.goalDict[goalWhichBoundV].rootAncestors:
                            gin.rootAncestors.add(a)
                    else:
                        gin.outputs.add(v)
                        vin.outputOf = j
            # if no ancestors then this is a root
            if not gin.rootAncestors: gin.rootAncestors.add(j)

        #validate - lhs has exactly one output, which is either bound
        #by a RHS goal or else is a constant
        lhsGin = self.goalDict[-1]
        assert len(lhsGin.outputs)==1, 'lhs must have exactly one output'        
        y = only(lhsGin.outputs)
        yVin = self.varDict[y]
        assert yVin.outputOf!=None or not parser.isVariableAtom(y), 'lhs output variable "%s" not bound' % y

    def removeIncomingChains(self):
        """Identify the goals which produce input variables other than the
        inputs from the lhs.  Example: in a clause like predict(X,Y)
        :- validClass(Y),classify(X,Y) the variable Y is bound by the
        validClass/1 predicate, not by the lhs.  We can't implement
        such a RHS with matrix multiplications, so instead we will
        create a conjunction of goals which are part of incoming
        chains, and explicitly find all tuples that satisfy these
        goals with a 'real' (wam-based) theorem-prover, and then
        finally remove these variables by binding them to constants in
        multiple copies of the original rule: eg for the example
        predict(X,Y):- validClass(Y),classify(X,Y) produce a rule
        predict(X,y):- classify(X,y) for each y:validClass(Y).
        """
        newRhs = []
        self.incoming = []
        self.explicitlyQuantified = set()
        for j in range(len(self.rule.rhs)):
            goal = self.rule.rhs[j]
            gin = self.goalDict[j]
            if -1 in gin.rootAncestors:
                newRhs.append(goal)
            else:
                self.incoming.append(goal)
                for v in gin.outputs:
                    self.explicitlyQuantified.add(v)
        # TODO skolem constants - this is readable, but needs to
        #change so I can substitute in different values later
        def skolemizeVar(v): 
            return ('sk$'+v) if v in self.explicitlyQuantified else v
        def skolemizeGoal(g):
            return parser.Goal(g.functor, map(skolemizeVar,g.args))
        # replace the variables produced by the incoming chain with
        # skolem constants 
        self.rule = parser.Rule(skolemizeGoal(self.rule.lhs), map(skolemizeGoal,newRhs))

    def compileDefinedPredicates(self):
        """Recursively call the tensorlog program to compile
        each subpredicate."""
        for j in range(len(self.rule.rhs)):
            gin = self.goalDict[j]
            mode = self.toMode(j)        
            if self.tensorlogProg.findPredDef(mode):
                gin.definedPred = True
                self.tensorlogProg.compile(mode,self.depth+1)

    def findMainChain(self):
        """Mark the goals on the 'main chain' from the lhs input to lhs output variables."""
        ginLHS = self.goalDict[-1]
        if ginLHS.outputs:
            v = only(ginLHS.outputs)
            #find where the output variable is bound
            j = self.varDict[v].outputOf
            #trace back to the lhs input.  if there are multiple
            #inputs to a goal, then find the one which depends on the
            #lhs input and follow that
            while j>=0:
                gin = self.goalDict[j]
                gin.mainChain = True
                # TODO can there be multiple inputs?
                v = only(gin.inputs)
                j = self.varDict[v].outputOf

    def generateOps(self):
        """ Convert the compiled rule to a sequence of matrix ops.
        """
        numMainChainGoals = len([j for j in range(len(self.rule.rhs)) if self.goalDict[j].mainChain])
        if not numMainChainGoals:
            lhsGin = self.goalDict[-1]            
            #find constraints on the input
            #TODO can we have multiple inputs in this case?
            x0 = only(lhsGin.inputs)
            self.detectSecondaryChainConstraints(x0)
            x = self.varDict[x0].constrainedVersion
            #generate code to weight the onehot vector for the
            #constant output by the score of the input
            constOut = only(lhsGin.outputs)
            tmpOut = 'weight_%s_by_%s' % (constOut,x0)
            self.varDict[constOut].constrainedVersion = tmpOut
            self.ops.append(ops.WeightedOnehot(tmpOut,x,constOut))

        #execute this code for mainChain goals
        for j in range(len(self.rule.rhs)):
            gin = self.goalDict[j]
            if gin.mainChain:
                #TODO compile pair predicates?
                x0 = only(gin.inputs)
                self.detectSecondaryChainConstraints(x0)
                x = self.varDict[x0].constrainedVersion
                y = only(gin.outputs)
                mode = self.toMode(j)
                if not gin.definedPred:
                    self.ops.append(ops.RightMatMulOp(y,x,mode))
                else:
                    self.ops.append(ops.DefinedPredOp(self.tensorlogProg,y,x,mode,self.depth+1))

        # also look for non-mainchain constraints on the final output
        # of the chain
        for v,vin in self.varDict.items():
            if (-1 in vin.inputTo) and parser.isVariableAtom(v):
                self.detectSecondaryChainConstraints(v)

    def detectSecondaryChainConstraints(self,x):
        """Append to self.ops operations that compute a variant of x which is
        constrained by 'backward' propagations from non-mainchain
        chains which use x as an input, and leave the result in a temp
        var v, and save v as varDict[x].constrainedVersion."""
        vin = self.varDict[x]
        if not vin.constraintsDone:
            #goals off the mainchain that use x as an input
            secondaryConsumers = [j for j in vin.inputTo if j>=0 and not self.goalDict[j].mainChain]
            #conjoin x with the product of all the 'backward' variables of secondary chains which use x as an input
            lastIntermediateProduct = x
            for j in secondaryConsumers:
                nextIntermediateProduct = "tmpProd_%s_%d" % (x,j)
                constaint_on_x_from_j = self.secondaryChainConstraint(x,j)
                self.ops.append(ops.ComponentwiseMulRowToColOp(nextIntermediateProduct,lastIntermediateProduct,constaint_on_x_from_j))
                lastIntermediateProduct = nextIntermediateProduct
            vin.constrainedVersion = lastIntermediateProduct
            # set flag so we don't do all this again
            vin.constraintsDone = True  
    
    # \vek{a}_j in the docs
    def secondaryChainConstraint(self,x,j):
        """Append to self.ops operations that compute a variant of x which is
        constrained by 'backward' propagations from the non-mainchain
        chains starting with goal j, and which must use x as an input.
        Leave the result in a temp var v, returning that temp variable
        name."""
        gin = self.goalDict[j]
        preimage_of_j = "tmpCon_%d" % j
        mode = self.toMode(j)
        if not gin.outputs:
            self.ops.append(ops.AssignPreimageToVar(preimage_of_j,mode))
        else:
            y0 = only(gin.outputs)
            vin = self.varDict[y0]
            if not vin.inputTo:
                #y0 is a free variable, not used anywhere else
                self.ops.append(ops.AssignPreimageToVar(preimage_of_j,mode))
            else:
                self.detectSecondaryChainConstraints(y0)
                y = vin.constrainedVersion
                self.ops.append(ops.LeftMatMulOp(preimage_of_j,y,mode))
        return preimage_of_j
                

    def toMode(self,j):
        """Return a mode declaration for the j-th goal of the rule"""
        goal = self.rule.rhs[j]
        def argIOMode(x): 
            return 'i' if x in self.goalDict[j].inputs else 'o'
        def argOnehotMode(x): 
            if (not parser.isVariableAtom(x)) or self.varDict[x].onehot: return '1'
            else: return ''
        return tensorlog.ModeDeclaration(parser.Goal(goal.functor, [argIOMode(x)+argOnehotMode(x) for x in goal.args]))

    #
    # access the result of compilation
    # 

    def getOps(self): 
        """ After compilation, return the operator sequence 
        """
        return self.ops

    def getInputs(self): 
        """ After compilation, return a list of input variables, which should
        be bound in the environment before eval-ing the ops.
        """
        return [v for v,vin in self.varDict.items() if vin.outputOf==-1]

    def getOutputs(self): 
        """ After compilation, return a list of output variables, which hold
        the final results
        """
        return [vin.constrainedVersion for v,vin in self.varDict.items() if -1 in vin.inputTo]


    #
    # debugging tools
    #

    def showVars(self):
        print "\t".join("var outOf onehot inputTo outOf inputTo revB".split())
        def _gs(j): 
            if j<0: return str(self.rule.lhs)
            else: return str(self.rule.rhs[j])
        for v in sorted(self.varDict.keys()):
            vin = self.varDict[v]
            print "\t".join([ v, str(vin.outputOf), str(vin.onehot), ",".join(map(str,vin.inputTo)), 
                              _gs(vin.outputOf), ",".join(map(_gs,vin.inputTo)),  vin.constrainedVersion])

    def showRule(self):
        #print "\t".join("id goal ins outs roots".split())
        print 'for all',self.explicitlyQuantified,'in',map(str,self.incoming),'...'
        print '-1\t',self.rule.lhs,':-\t',str(self.goalDict[-1])
        for j in range(len(self.rule.rhs)):
            print '%2d' % j,'\t ',self.rule.rhs[j],'\t',str(self.goalDict[j])

    def showOps(self):
        print 'inputs:',",".join(self.getInputs())
        print 'outputs:',",".join(self.getOutputs())
        print 'compiled to:'
        for op in self.ops:
            print '\t',op

#
# a test driver
#

if __name__ == "__main__":
    if len(sys.argv)<2:
        print 'usage: rule mode'
        sys.exit(-1)

    STRICT = False
    p = parser.Parser()

    ruleString = sys.argv[1]
    rule = p.parseRule(ruleString)

    mode = tensorlog.ModeDeclaration(sys.argv[2])

    rules = parser.RuleCollection()
    rules.add(rule)
    prog = tensorlog.Program(db=None,rules=rules)
    c = OpCompiler(prog,0,rule)
    c.compile(mode)
    c.showRule()
    c.showVars()
    c.showOps()
    print 'incoming',map(str,c.incoming)
