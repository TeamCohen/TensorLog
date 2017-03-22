# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# top-level constructs for Tensorlog - Programs and Interpreters
#

import sys
import logging
import collections
import numpy as np

from tensorlog import bpcompiler
from tensorlog import comline
from tensorlog import declare
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import mutil
from tensorlog import opfunutil
from tensorlog import parser

VERSION = '1.3.1'

# externally visible changes:
#
# version 1.0: refactored cleaned-up version of nips-submission codebase
# version 1.1: thread-safe gradient and eval computation, and a parallel learner
# version 1.1.1: cleaned up trace function api, bug fix in parallel learner
# version 1.1.2: tracer output is not per example, no parallel option in funs
# version 1.2.0: not sure, really.
# version 1.2.1: plearn replaces epoch-level status monitoring with merged results minibatches
# version 1.2.2: add learner option to expt command-line
# version 1.2.3:
#    add p(X,Y) :- ... {foo(F): q(X,F)} templates, propprProg.setRuleWeights(), propprProg.setFeatureWeights()
#    list --prog xxx --ruleids
#    more options for expt
# version 1.2.4:
#    added --params and --weightEpsilon to expt.py
#    made conf.ops.long_trace a number
#    added interp.set()
# version 1.2.5:
#    cross-compilation
# version 1.3.0:
#    tensorlog is a module
#    new api for cross compilers + "simple" api
#    type declaration in cfacts: # :- actedIn(actor_t,movie_t)
#    parameter declarations:     # :- trainable(posWeight,1)
#    OOV marker for test/train .exam files
#    interp.Interp split off from program
# version 1.3.1:
#     simple.Compiler() fleshed out and tested for tensorflow
# version 1.3.2:
#     AbstractCrossCompiler.possibleOps() added

DEFAULT_MAXDEPTH=10
DEFAULT_NORMALIZE='softmax'
#DEFAULT_NORMALIZE='log+softmax' # equiv to 'ordinary' normalization
#DEFAULT_NORMALIZE='none'        # don't normalize

##############################################################################
## a program
##############################################################################

class Program(object):

    def __init__(self, db=None, rules=parser.RuleCollection(), calledFromProPPRProgram=False):
        self.db = db
        self.function = {}
        self.rules = rules
        self.maxDepth = DEFAULT_MAXDEPTH
        self.normalize = DEFAULT_NORMALIZE
        # check the rules aren't proppr formatted
        def checkRule(r):
            assert not r.features, 'for rules with {} features, specify --proppr: %s' % str(r)
            return r
        if not calledFromProPPRProgram:
            self.rules.mapRules(checkRule)

    def clearFunctionCache(self):
        self.function = {}

    def findPredDef(self,mode):
        """Find the set of rules with a lhs that match the given mode."""
        return self.rules.rulesFor(mode)

    def compile(self,mode,depth=0):
        """ Produce an funs.Function object which implements the predicate definition
        """
        #find the rules which define this predicate/function

        if (mode,depth) in self.function:
            return self.function[(mode,depth)]

        if depth>self.maxDepth:
            self.function[(mode,depth)] = funs.NullFunction(mode)
        else:
            predDef = self.findPredDef(mode)
            if len(predDef)==0:
                assert False,'no rules match mode %s' % mode
            elif len(predDef)==1:
                #instead of a sum of one function, just find the function
                #for this single predicate
                c = bpcompiler.BPCompiler(mode,self,depth,predDef[0])
                self.function[(mode,depth)] = c.getFunction()
            else:
                #compute a function that will sum up the values of the
                #clauses
                ruleFuns = map(lambda r:bpcompiler.BPCompiler(mode,self,depth,r).getFunction(),predDef)
                self.function[(mode,depth)] = funs.SumFunction(ruleFuns)
            if depth==0:
                if self.normalize=='softmax':
                    self.function[(mode,0)] = funs.SoftmaxFunction(self.function[(mode,0)])
                elif self.normalize=='log+softmax':
                    self.function[(mode,0)] = funs.SoftmaxFunction(funs.LogFunction(self.function[(mode,0)]))
                elif self.normalize=='none':
                    pass
                else:
                    assert not self.normalize, 'bad value of self.normalize: %r' % self.normalize
                # label internal nodes/ops of function with ids
                self.function[(mode,0)].install()
        return self.function[(mode,depth)]

    def getPredictFunction(self,mode):
        if (mode,0) not in self.function: self.compile(mode)
        fun = self.function[(mode,0)]
        return fun

    def getParamList(self):
        """ Return a set of (functor,arity) pairs corresponding to the parameters """
        return self.db.paramList

    def getFunction(self,mode):
        """ Return the compiled function for a mode """
        if (mode,0) not in self.function: self.compile(mode)
        return self.function[(mode,0)]

    def evalSymbols(self,mode,symbols,typeName=None):
        """ After compilation, evaluate a function.  Input is a list of
        symbols that will be converted to onehot vectors, and bound to
        the corresponding input arguments.
        """
        return self.eval(mode, [self.db.onehot(s,typeName=typeName) for s in symbols])

    def eval(self,mode,inputs):
        """ After compilation, evaluate a function.  Input is a list of onehot
        vectors, which will be bound to the corresponding input
        arguments.
        """
        if (mode,0) not in self.function: self.compile(mode)
        fun = self.function[(mode,0)]
        return fun.eval(self.db, inputs, opfunutil.Scratchpad())

    def evalGradSymbols(self,mode,symbols):
        """ After compilation, evaluate a function.  Input is a list of
        symbols that will be converted to onehot vectors, and bound to
        the corresponding input arguments.
        """
        assert self.db.isTypeless(),'cannot evalSymbols on db with declared types'
        return self.evalGrad(mode, [self.db.onehot(s) for s in symbols])

    def evalGrad(self,mode,inputs):
        """ After compilation, evaluate a function.  Input is a list of onehot
        vectors, which will be bound to the corresponding input
        arguments.
        """
        if (mode,0) not in self.function: self.compile(mode)
        fun = self.function[(mode,0)]
        return fun.evalGrad(self.db, inputs)

    def setAllWeights(self):
        """ Set all parameter weights to a plausible value - mostly useful for proppr programs,
        where parameters are known. """
        logging.debug('setting feature weights %.3f Gb' % comline.memusage())
        self.setFeatureWeights()
        logging.debug('setting rule weights %.3f Gb' % comline.memusage())
        self.setRuleWeights()
        self.db.checkTyping()

    def setFeatureWeights(self,epsilon=1.0):
        """ Set feature weights to a plausible value - mostly useful for proppr programs,
        where parameters are known. """
        logging.warn('trying to call setFeatureWeights on a non-ProPPR program')

    def setRuleWeights(self,weights=None,epsilon=1.0):
        """ Set rule feature weights to a plausible value - mostly useful for proppr programs,
        where parameters are known. """
        logging.warn('trying to call setFeatureWeights on a non-ProPPR program')

    @staticmethod
    def _load(fileNames,db=None):
        ruleFiles = [f for f in fileNames if f.endswith(".ppr") or f.endswith(".tlog")]
        dbFiles = [f for f in fileNames if f.endswith(".db")]
        factFiles = [f for f in fileNames if f.endswith(".cfacts")]
        assert (not dbFiles) or (not factFiles), 'cannot combine a serialized database and .cfacts files'
        assert (not dbFiles) or (len(dbFiles)==1), 'cannot combine multiple serialized databases'
        assert db or dbFiles or factFiles,'no db specified'
        assert ruleFiles,'no rules specified'
        rules = parser.Parser.parseFile(ruleFiles[0])
        for f in ruleFiles[1:]:
            rules = parser.Parser.parseFile(f,rules)
        if dbFiles:
            db = matrixdb.MatrixDB.deserialize(dbFiles[0])
        if factFiles:
            db = matrixdb.MatrixDB()
            for f in factFiles:
                logging.debug("starting %s" % f)
                db.addFile(f)
                logging.debug("finished %s" % f)
        return (db,rules)

    @staticmethod
    #TODO: deprecate
    def load(fileNames,db=None):
        if not db: (db,rules) = Program._load(fileNames)
        else: (dummy,rules) = Program._load(fileNames,db=db)
        return Program(db=db,rules=rules)

    @staticmethod
    def _loadRules(fileNames):
        ruleFiles = fileNames.split(":")
        rules = parser.Parser.parseFile(ruleFiles[0])
        for f in ruleFiles[1:]:
            rules = parser.Parser.parseFile(f,rules)
        return rules

    @staticmethod
    def loadRules(fileNames,db):
        return Program(db,Program._loadRules(fileNames))


#
# subclass of Program that corresponds more or less to Proppr....
#

class ProPPRProgram(Program):

    def __init__(self, db=None, rules=parser.RuleCollection(), weights=None):
        super(ProPPRProgram,self).__init__(db=db, rules=rules, calledFromProPPRProgram=True)
        # dictionary mapping parameter name to list of modes that can
        # be used to determine possible non-zero values for the
        # parameters
        self.paramDomains = collections.defaultdict(list)
        # list of constants used as rule features
        self.ruleIds = []
        #expand the syntactic sugar used by ProPPR
        self.rules.mapRules(self._moveFeaturesToRHS)
        if weights!=None: self.setRuleWeights(weights)

    def setRuleWeights(self,weights=None,epsilon=1.0,ruleIdPred=None):
        """Set the db predicate 'weighted/1' as a parameter, and initialize it
        to the given vector.  If no vector 'weights' is given, default
        to a constant vector of epsilon for each rule.  'weighted/1'
        is the default parameter used to weight rule-ids features,
        e.g., "r" in p(X,Y):-... {r}.  You can also specify the
        ruleIds with the name of a unary db relation that holds all
        the rule ids.
        """
        if len(self.ruleIds)==0:
            logging.warn('no rule features have been defined')
        elif ruleIdPred is not None:
            # TODO check this stuff and add type inference!
            assert (ruleIdPred,1) in set.matEncoding,'there is no unary predicate called %s' % ruleIdPred
            self.db.markAsParameter("weighted",1)
            self.db.setParameter(self.vector(declare.asMode('%s(o)' % ruleIdPred)) * epsilon)
        else:
            assert self.db.isTypeless(), 'cannot setRuleWeights for db with declared types unless ruleIdPred is given'
            self.db.markAsParameter("weighted",1)
            if weights==None:
                weights = self.db.onehot(self.ruleIds[0])
                for rid in self.ruleIds[1:]:
                    weights = weights + self.db.onehot(rid)
                weights = mutil.mapData(lambda d:np.clip(d,0.0,1.0), weights)
            self.db.setParameter("weighted",1,weights*epsilon)

    def getRuleWeights(self):
        """ Return a vector of the weights for a rule """
        return self.db.matEncoding[('weighted',1)]

    def setFeatureWeights(self,epsilon=1.0):
        """Initialize each feature used in the feature part of a rule, i.e.,
        for all rules annotated by "{foo(F):...}", declare 'foo/1' to
        be a parameter, and initialize it to something plausible.  The
        'something plausible' is based on looking at how the variables
        defining foo are defined, eg for something like "p(X,Y):-
        ... {posWeight(F):hasWord(X,F)}" a constant sparse vector with
        non-zero weights for all second arguments of hasWord will be
        used to initialize posWeight.  The constant will be epsilon.
        """
        for paramName,domainModes in self.paramDomains.items():
            # we also need to infer a type for the parameter....
            def typeOfWeights(mode):
                for i in range(mode.arity):
                    if mode.isInput(i):
                        return self.db.getArgType(mode.functor,mode.arity,i)
                assert False
            weights = self.db.matrixPreimage(domainModes[0])
            weightType = typeOfWeights(domainModes[0])
            for mode in domainModes[1:]:
                weights = weights + self.db.matrixPreimage(mode)
                assert typeOfWeights(mode)==weightType, 'feature weights have incompatible types: derived from %s and %s' % (mode,domainModes[0])
            weights = weights * 1.0/len(domainModes)
            weights = mutil.mapData(lambda d:np.clip(d,0.0,1.0), weights)
            self.db.setParameter(paramName,1,weights*epsilon)
            decl = declare.TypeDeclaration(parser.Goal(paramName,[weightType]))
            self.db.addTypeDeclaration(decl,'<autoseting parameters>',-1)
            logging.debug('parameter %s/1 initialized to %s' % (paramName,"+".join(map(lambda dm:'preimage(%s)' % str(dm), domainModes))))
            logging.debug('type declaration for %s/1 is %s' % (paramName,decl))
        for (paramName,arity) in self.getParamList():
            if not self.db.parameterIsInitialized(paramName,arity):
                logging.warn("Parameter %s could not be set automatically")
        logging.debug('total parameter size: %d', self.db.parameterSize())

    def setFeatureWeight(self,paramName,arity,weight):
        """ Set a particular parameter weight. """
        self.db.markAsParameter(paramName,arity)
        self.db.setParameter(paramName,arity,weight)

    def _moveFeaturesToRHS(self,rule0):
        rule = parser.Rule(rule0.lhs, rule0.rhs)
        if not rule0.findall:
            #parsed format is {f1,f2,...} but we only support {f1}
            if rule0.features is None:
              logging.warn('this rule has no features: %s' % str(rule))
            else:
              assert len(rule0.features)==1,'multiple constant features not supported'
              assert rule0.features[0].arity==0, '{foo(A,...)} not allowed, use {foo(A,...):true}'
              constFeature = rule0.features[0].functor
              constAsVar = constFeature.upper()
              rule.rhs.append( parser.Goal(bpcompiler.ASSIGN, [constAsVar,constFeature]) )
              rule.rhs.append( parser.Goal('weighted',[constAsVar]) )
              # record the rule name, ie the constant feature
              self.ruleIds.append(constFeature)
        else:
            #format is {foo(F):-...}
            assert len(rule0.features)==1,'feature generators of the form {a,b: ... } not supported'
            featureLHS = rule0.features[0]
            assert featureLHS.arity==1, 'non-constant features must be of the form {foo(X):-...}'
            outputVar = featureLHS.args[0]
            paramName = featureLHS.functor
            for goal in rule0.findall:
                if goal.arity!=0 and goal.functor!='true':
                  rule.rhs.append(goal)
            rule.rhs.append( parser.Goal(paramName,[outputVar]) )
            # record the feature predicate 'foo' as a parameter
            if self.db: self.db.markAsParameter(paramName,1)
            # record the domain of the predicate
            for goal in rule0.findall:
                if outputVar in goal.args:
                    k = goal.args.index(outputVar)
                    if goal.arity==2:
                        paramMode = declare.asMode("%s/io" % goal.functor) if k==0 else declare.asMode("%s/oi" % goal.functor)
                        self.paramDomains[paramName].append(paramMode)
        return rule

    #TODO: deprecate
    @staticmethod
    def load(fileNames,db=None):
        if not db: (db,rules) = Program._load(fileNames)
        else: (dummy,rules) = Program._load(fileNames,db=db)
        return ProPPRProgram(db=db,rules=rules)

    @staticmethod
    def loadRules(fileNames,db):
        return ProPPRProgram(db=db,rules=Program._loadRules(fileNames))
