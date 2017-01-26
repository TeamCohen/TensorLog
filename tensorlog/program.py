# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# top-level constructs for Tensorlog - Programs and Interpreters
#

import sys
import logging
import collections

from tensorlog import bpcompiler
from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import opfunutil
from tensorlog import ops
from tensorlog import parser

# interactive debugger is sort of optional, and it depends on a bunch
# of stuff that might not be present, so don't require this import
try:
  from tensorlog import debug
  DEBUGGER_AVAILABLE = True
except ImportError:
  logging.warn('debug module not imported')
  DEBUGGER_AVAILABLE = False

VERSION = "1.2.5"

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
        """ Produce an ops.Function object which implements the predicate definition
        """
        #find the rules which define this predicate/function
        
        if (mode,depth) in self.function:
            return

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
                ruleFuns = map(lambda r:bpcompiler.BPCompiler(mode,self,depth,r).getFunction(), predDef)
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

    def evalSymbols(self,mode,symbols):
        """ After compilation, evaluate a function.  Input is a list of
        symbols that will be converted to onehot vectors, and bound to
        the corresponding input arguments.
        """
        return self.eval(mode, [self.db.onehot(s) for s in symbols])

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
        return self.evalGrad(mode, [self.db.onehot(s) for s in symbols])

    def evalGrad(self,mode,inputs):
        """ After compilation, evaluate a function.  Input is a list of onehot
        vectors, which will be bound to the corresponding input
        arguments.
        """
        if (mode,0) not in self.function: self.compile(mode)
        fun = self.function[(mode,0)]
        return fun.evalGrad(self.db, inputs)

    def setFeatureWeights(self,epsilon=1.0):
        logging.warn('trying to call setFeatureWeights on a non-ProPPR program')

    def setRuleWeights(self,weights=None,epsilon=1.0):
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

    def setRuleWeights(self,weights=None,epsilon=1.0):
        """Set the db predicate 'weighted/1' as a parameter, and initialize it
        to the given vector.  If no vector is given, default to a
        sparse vector of all constant rule features. 'weighted/1' is
        the default parameter used to weight rule-ids features, e.g.,
        "r" in p(X,Y):-... {r}.
        """
        if len(self.ruleIds)==0: 
            logging.warn('no rule features have been defined')
        else:
            self.db.markAsParam("weighted",1)
            if weights==None:
                weights = self.db.onehot(self.ruleIds[0])
                for rid in self.ruleIds[1:]:
                    weights = weights + self.db.onehot(rid)
            self.db.setParameter("weighted",1,weights*epsilon)

    def getRuleWeights(self):  
        """ Return a vector of the weights for a rule """
        return self.db.matEncoding[('weighted',1)]

    def setFeatureWeights(self,epsilon=1.0):
        """ Initialize each feature used in the feature part of a rule, i.e.,
        for all rules annotated by "{foo(F):...}", declare 'foo/1' to
        be a parameter, and initialize it to something plausible.  The
        'something plausible' is based on looking at how the variables
        defining foo are defined, eg for something like "p(X,Y):-
        ... {posWeight(F):hasWord(X,F)}" the sparse vector of all
        second arguments of hasWord will be used to initialize
        posWeight.
        """
        for paramName,domainModes in self.paramDomains.items():
            weights = self.db.matrixPreimage(domainModes[0])
            for mode in domainModes[1:]:
                weights = weights + self.db.matrixPreimage(mode)
            weights = weights * 1.0/len(domainModes)
            self.db.setParameter(paramName,1,weights*epsilon)
            logging.info('parameter %s/1 initialized to %s' % (paramName,"+".join(map(lambda dm:'preimage(%s)' % str(dm), domainModes))))
        for (paramName,arity) in self.getParams():
            if not self.db.parameterIsSet(paramName,arity):
                logging.warn("Parameter %s could not be set automatically")
        logging.info('total parameter size: %d', self.db.parameterSize())

    def setFeatureWeight(self,paramName,arity,weight):
        """ Set a particular parameter weight. """
        self.db.markAsParam(paramName,arity)
        self.db.setParameter(paramName,arity,weight)

    def getParams(self):
        """ Return a set of (functor,arity) pairs corresponding to the parameters """
        return self.db.params

    def _moveFeaturesToRHS(self,rule0):
        rule = parser.Rule(rule0.lhs, rule0.rhs)
        if not rule0.findall:
            #parsed format is {f1,f2,...} but we only support {f1}
            assert len(rule0.features)==1,'multiple constant features not supported'
            constFeature = rule0.features[0].functor
            constAsVar = constFeature.upper()
            rule.rhs.append( matrixdb.assignGoal(constAsVar,constFeature) )
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
                rule.rhs.append(goal)
            rule.rhs.append( parser.Goal(paramName,[outputVar]) )
            # record the feature predicate 'foo' as a parameter
            if self.db: self.db.markAsParam(paramName,1)
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

class Interp(object):
    """High-level interface to tensor log."""

    def __init__(self,prog,trainData=None,testData=None):
        self.prog = prog
        self.db = self.prog.db
        self.trainData = trainData
        self.testData = testData
        self.numTopEcho = 10

    def help(self):
        print "ti.list(foo): foo can be a compiled function, eg \"foo/io\", a predicate definition, eg"
        print "              \"foo/2\", or a database predicate, also specified as \"foo/2\"."
        print "ti.list():    list everything."
        print "ti.eval(\"functor/mode\",\"c\"): evaluate a function on a database constant c"
        if DEBUGGER_AVAILABLE:
          print "ti.debug(\"functor/mode\",\"c\"): debug the corresponding eval command"
          print "ti.debugDset(\"functor/mode\"[,test=True]): run debugger on a dataset"
        print "ti.set(depth=d): reset the max depth"
        print "ti.set(normalize='none'|'softmax'|'log+softmax'): reset the normalizer"
        print "ti.set(echo=N): number of top items to print when doing an eval"
        print "ti.set(trace=True|False): print summary of messages sent in inference"
        print "ti.set(maxTraceMsg=N): print data in messages if they have fewer than N non-zero entries"

    def set(self,depth=None,echo=None,normalize=None,maxTraceMsg=-1,trace=None):
        if depth!=None:
            self.prog.maxDepth = depth
            self.prog.clearFunctionCache()
        if normalize!=None:
            self.prog.normalize = normalize
            self.prog.clearFunctionCache()
        if echo!=None:
            self.numTopEcho = echo
        if trace!=None:
            ops.conf.trace = trace
        if maxTraceMsg>=0:
            ops.conf.long_trace = maxTraceMsg

    def list(self,str=None):
        if str==None:
            self._listAllRules()
            self._listAllFacts()
        else:
            assert str.find("/")>=0, 'supported formats are functor/arity, function/io, function/oi, function/o, function/i'
            functor,rest = str.split("/")
            try:
                arity = int(rest)
                self._listRules(functor,arity) or self._listFacts(functor,arity)
            except ValueError:
                self._listFunction(str)

    def _listRules(self,functor,arity):
        mode = declare.ModeDeclaration(parser.Goal(functor,['x']*arity),strict=False)
        rules = self.prog.rules.rulesFor(mode)
        if rules:
            for r in rules: print r
            return True
        return False

    def _listAllRules(self):
        self.prog.rules.listing()

    def _listFacts(self,functor,arity):
        if self.db.inDB(functor,arity):
            print self.db.summary(functor,arity)
            return True
        return False

    def _listAllFacts(self):
        self.db.listing()

    def _listFunction(self,modeSpec):
        mode = declare.asMode(modeSpec)
        key = (mode,0)
        if key not in self.prog.function:
            self.prog.compile(mode)
        fun = self.prog.function[key]
        print "\n".join(fun.pprint())

    def eval(self,modeSpec,sym):
        mode = declare.asMode(modeSpec)
        tmp = self.prog.evalSymbols(mode,[sym])
        result = self.prog.db.rowAsSymbolDict(tmp)
        if (self.numTopEcho):
            top = sorted(map(lambda (key,val):(val,key), result.items()), reverse=True)
            for rank in range(min(len(top),self.numTopEcho)):
                print '%d\t%g\t%s' % (rank+1,top[rank][0],top[rank][1])
        return result

    def debug(self,modeSpec,sym):
      if not DEBUGGER_AVAILABLE:
        logging.warn('debugger is not available in this environment')
        return
      mode = declare.asMode(modeSpec)
      X = self.db.onehot(sym)
      dset = dataset.Dataset({mode:X},{mode:self.db.zeros()})
      debug.Debugger(self.prog,mode,dset,gradient=False).mainloop()

    def debugDset(self,modeSpec,test=False):
      if not DEBUGGER_AVAILABLE:
        logging.warn('debugger is not available in this environment')
        return
      fullDataset = self.testData if test else self.trainData
      if fullDataset==None:
        print 'train/test dataset is not specified on command line?'
      else:
        mode = declare.asMode(modeSpec)
        dset = fullDataset.extractMode(mode)
        debug.Debugger(self.prog,mode,dset,gradient=True).mainloop()

#
# sample main: python tensorlog.py test/fam.cfacts 'rel(i,o)' 'rel(X,Y):-spouse(X,Y).' william
#

if __name__ == "__main__":

    print "Tensorlog v%s (C) William W. Cohen and Carnegie Mellon University, 2016" % VERSION

    optdict,args = comline.parseCommandLine(sys.argv[1:])
    ti = Interp(optdict['prog'],trainData=optdict.get('trainData'),testData=optdict.get('testData'))

    try:
        if sys.ps1: interactiveMode = True
    except AttributeError:
        interactiveMode = False
        if sys.flags.interactive: interactiveMode = True
    if interactiveMode:
        print "Interpreter variable 'ti' set, type ti.help() for help"
    else:
        print "Usage: python -i -m tensorlog [opts]"
        print "- For option help: 'python -m tensorlog --help'"
        print "- The interpreter is really only useful with the -i option."
