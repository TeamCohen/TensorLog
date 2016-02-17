# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import ops
import parser
import matrixdb
import bpcompiler

##############################################################################
## declarations
##############################################################################

MAXDEPTH=10

class AbstractDeclaration(object):
    def __init__(self,goal):
        if type(goal)==type(""):
            goal = parser.Parser.parseGoal(goal)
        self.prototype = goal
        self._key = str(goal)
    def arg(self,i):
        return self.prototype.args[i]
    def getArity(self):
        return self.prototype.arity
    def getFunctor(self):
        return self.prototype.functor
    arity = property(getArity)
    functor = property(getFunctor)
    def __str__(self):
        return str(self.prototype)
    def __repr__(self):
        return repr(self.prototype)
    def __hash__(self):
        return hash(self._key)
    def __eq__(self,other):
        return other and isinstance(other,AbstractDeclaration) and self._key == other._key

class ModeDeclaration(AbstractDeclaration):
    """Declare a mode with a goal, eg hasWord(i1,o).  Arguments starting
    with 'i' (respectively 'o') are inputs (outputs), and arguments
    ending with '1' are one-hot encodings, aka singleton sets.
    """
    def isInput(self,i):
        return self.arg(i)=='i'
    def isOutput(self,i):
        return self.arg(i)=='o'
    def isConst(self,i):
        return not self.isInput(i) and not self.isOutput(i)

#TODO unused, should I delete?

class TypeDeclaration(AbstractDeclaration):
    """Declare predicate types with a goal, eg hasWord(doc,word).
    """
    def type(self,i):
        self.arg(i)


##############################################################################
## a program
##############################################################################


class Program(object):

    def __init__(self, db=None, rules=parser.RuleCollection()):
        self.db = db
        self.program = []
        self.function = {}
        self.rules = rules

    def findPredDef(self,mode):
        """Find the set of rules with a lhs that match the given mode."""
        return self.rules.rulesFor(mode)

    def compile(self,mode,depth=0):
        """ Produce an ops.Function object which implements the predicate definition
        """
        #find the rules which define this predicate/function
        
        if depth>MAXDEPTH:
            nullFun = bpcompiler.buildNullFunction(mode)
            self.function[(mode,depth)] = nullFun
        else:
            predDef = self.findPredDef(mode)
            if len(predDef)==0:
                assert False,'no rules match mode %s' % mode
            elif len(predDef)==1:
                #instead of a sum of one function, just find the function
                #for this single predicate
                c = bpcompiler.BPCompiler(self,depth,predDef[0])
                #print '= calling bp.compile',mode,'depth',depth
                c.compile(mode)
                self.function[(mode,depth)] = ops.OpFunction(c.getInputs(), c.getOutputs(), ops.SeqOp(c.getOps()))            
            else:
                #compute a function that will sum up the values of the
                #clauses
                ruleFuns = []
                for r in predDef:
                    c = bpcompiler.BPCompiler(self,depth,r)
                    c.compile(mode)
                    ruleFuns.append( ops.OpFunction(c.getInputs(),c.getOutputs(),ops.SeqOp(c.getOps())) )
                self.function[(mode,depth)] = ops.SumFunction(ruleFuns)
        return self.function[(mode,depth)]

    def functionListing(self):
        for (m,d) in sorted(self.function.keys()):
            print '> mode',m,'depth',d,'fun:',self.function[(m,d)]

    def eval(self,mode,symbols):
        """ After compilation, evaluate a function.  Input is a list of symbols
        that will be converted to onehot vectors.
        """
        fun = self.function[(mode,0)]
        return fun.eval(self.db, [self.db.onehot(s) for s in symbols])

    def theanoPredictFunction(self,mode,symbols):
        """ After compilation, produce a theano function f which computes the
        appropriate output values, by delegation to the appropriate
        compiled ops.Function object. To evaluate f, call
        f(x1,...,xk) where xi's are onehot representations.
        """
        fun = self.function[(mode,0)]
        return fun.theanoPredictFunction(self.db, symbols)

# TODO remove?
#
# subclass of Program that corresponds more or less to Proppr....
# 

class ProPPRProgram(Program):

    def __init__(self, db=None, rules=parser.RuleCollection(),weights=None):
        super(ProPPRProgram,self).__init__(db=db, rules=rules)
        self.weights = ops.columnVectorParam(weights,"weights")
        
    def ruleListing(self,mode):
        for r in self.findPredDef(mode):
            print 'rule:',str(r)
            print 'generator:',self._featureGenerator(r,mode)

    def compile(self,mode,depth=0):
        """ Produce an ops.Function object which implements the predicate definition
        """
        if depth>MAXDEPTH:
            nullFun = bpcompiler.buildNullFunction(mode)
            self.function[(mode,depth)] = nullFun
        else:
            predDef = self.findPredDef(mode)
            assert predDef,'no rules match mode %s' % mode
            # We need to adjust the function appropriately for each
            # rule r, using the special WeightedSumFunction which
            # takes two parallel lists of functions and accumulates
            # them appropriately.
            ruleFuns = []
            featureFuns = []
            for r in predDef:
                cr = bpcompiler.BPCompiler(self,depth,r)
                cr.compile(mode)
                ruleFuns.append( ops.OpFunction(cr.getInputs(),cr.getOutputs(),ops.SeqOp(cr.getOps())) )
                g = self._featureGenerator(r,mode)
                cg = bpcompiler.BPCompiler(self,depth,g)
                cg.compile(mode)
                featureFuns.append( ops.OpFunction(cg.getInputs(),cg.getOutputs(),ops.SeqOp(cg.getOps())) )
            self.function[(mode,depth)] = ops.WeightedSumFunction(self.weights,ruleFuns,featureFuns)
        return self.function[(mode,depth)]

    def _featureGenerator(self,rule,mode):
        # given a rule with features produce a new rule that will
        # generate the appropriate weighted set of features.  For
        # example, for "p(X,Y) :- q(X,Y) { f0 }." and mode p(i,o) this
        # will generate a rule "feature__(X,f0):-." and for "p(X,Y) :-
        # q(X,Y) { all(F):r(X,F) }." and mode p(i,o) this will
        # generate a rule "feature__(X,F):-r(X,F)".  You can replace
        # q(X,Y) and r(X,F) above with any legal body.
        inputVarPositions = [i for i in range(rule.lhs.arity) if mode.isInput(i)]
        assert len(inputVarPositions)==1
        inputVar = rule.lhs.args[inputVarPositions[0]]
        if not rule.findall:
            #format is {f1,f2,...} but so far I only support {f1}
            assert len(rule.features)==1,'multiple constant features not supported'
            constFeature = rule.features[0].functor
            return parser.Rule(parser.Goal('feature__',[inputVar,constFeature]),[])
        else:
            #format is {all(F):-...}
            assert len(rule.features)==1,'feature generators of the form {a,b: ... } not supported'
            featureLHS = rule.features[0]
            assert featureLHS.arity==1 and featureLHS.functor=='all', 'non-constant features must be of the form {all(X):-...}'
            outputVar = featureLHS.args[0] 
            return parser.Rule(parser.Goal('feature__',[inputVar,outputVar]),rule.findall)

#
# sample main: python tensorlog.py test/fam.cfacts 'rel(i,o)' 'rel(X,Y):-spouse(X,Y).' william
#
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'usage factfile rule1 ... mode x1 ...'
    else:
        db = matrixdb.MatrixDB.loadFile(sys.argv[1])
        mode = None
        rules = parser.RuleCollection()
        xs = []
        for a in sys.argv[2:]:
            if a.find(":-") >= 0:
                rules.add(parser.Parser.parseRule(a))
            elif a.find("(") >=0:
                assert mode==None, 'only one mode allowed'
                mode = ModeDeclaration(a)
            else:
                xs.append(a)
        p = Program(db=db,rules=rules)
        p.functionListing()
        assert mode,'mode must be defined'
        f = p.compile(mode)
        p.functionListing()

        for x in xs:
            print 'native result on input "%s":' % x
            result = p.eval(mode,[x])
            for val in result:
                print db.rowAsSymbolDict(val)
            print 'theano result on input "%s":' % x
            f = p.theanoPredictFunction(mode,['x'])
            result = f(db.onehot(x))
            for val in result:            
                #print val
                print db.rowAsSymbolDict(val)

