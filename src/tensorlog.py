# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# top-level constructs for Tensorlog - Programs and Interpreters
#

import sys
import logging
import getopt

import declare
import funs
import ops
import parser
import matrixdb
import bpcompiler
import dataset
import learn
import mutil
import debug

VERSION = "1.0.02"

DEFAULT_MAXDEPTH=10
DEFAULT_NORMALIZE='softmax'
#DEFAULT_NORMALIZE='log+softmax'

##############################################################################
## a program
##############################################################################

class Program(object):

    def __init__(self, db=None, rules=parser.RuleCollection()):
        self.db = db
        self.function = {}
        self.rules = rules
        self.maxDepth = DEFAULT_MAXDEPTH
        self.normalize = DEFAULT_NORMALIZE

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
        return fun.eval(self.db, inputs, ops.Scratchpad())

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
        super(ProPPRProgram,self).__init__(db=db, rules=rules)
        #expand the syntactic sugar used by ProPPR
        self.rules.mapRules(ProPPRProgram._moveFeaturesToRHS)
        if weights!=None: self.setWeights(weights)

    def setWeights(self,weights):
        self.db.markAsParam("weighted",1)
        self.db.setParameter("weighted",1,weights)

    def getWeights(self):  
        return self.db.matEncoding[('weighted',1)]

    def getParams(self):
        return self.db.params

    @staticmethod
    def _moveFeaturesToRHS(rule0):
        rule = parser.Rule(rule0.lhs, rule0.rhs)
        if not rule0.findall:
            #parsed format is {f1,f2,...} but we only support {f1}
            assert len(rule0.features)==1,'multiple constant features not supported'
            constFeature = rule0.features[0].functor
            constAsVar = constFeature.upper()
            rule.rhs.append( matrixdb.assignGoal(constAsVar,constFeature) )
            rule.rhs.append( parser.Goal('weighted',[constAsVar]) )
        else:
            #format is {all(F):-...}
            assert len(rule0.features)==1,'feature generators of the form {a,b: ... } not supported'
            featureLHS = rule0.features[0]
            assert featureLHS.arity==1 and featureLHS.functor=='all', 'non-constant features must be of the form {all(X):-...}'
            outputVar = featureLHS.args[0] 
            for goal in rule0.findall:
                rule.rhs.append(goal)
            rule.rhs.append( parser.Goal('weighted',[outputVar]) )
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

    def help(self):
        print "ti.list(foo): foo can be a compiled function, eg \"foo/io\", a predicate definition, eg"
        print "              \"foo/2\", or a database predicate, also specified as \"foo/2\"."
        print "ti.list():    list everything."
        print "ti.eval(\"functor/mode\",\"c\"): evaluate a function on a database constant c"
        print "ti.debug(\"functor/mode\",\"c\"): debug the corresponding eval command"
        print "ti.debugDset(\"functor/mode\"[,test=True]): run debugger on a dataset"

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
        result = self.prog.evalSymbols(mode,[sym])
        return self.prog.db.rowAsSymbolDict(result)
    
    def debug(self,modeSpec,sym):
        mode = declare.asMode(modeSpec)        
        X = self.db.onehot(sym)
        dset = dataset.Dataset({mode:X},{mode:self.db.zeros()})
        debug.Debugger(self.prog,mode,dset,gradient=False).mainloop()
    
    def debugDset(self,modeSpec,test=False):
        fullDataset = self.testData if test else self.trainData
        if fullDataset==None:
            print 'train/test dataset is not specified on command line?'
        else:
            mode = declare.asMode(modeSpec)        
            dset = fullDataset.extractMode(mode)
            debug.Debugger(self.prog,mode,dset,gradient=True).mainloop()

#
# utilities for reading command lines
#

def parseCommandLine(argv):
    """
    See the usage() subfunction for the options that are parsed().
    Returns a dictionary mapping option names to Python objects,
    eg Datasets, Programs, ...
    """

    argspec = ["db=", "proppr", "prog=", "trainData=", "testData=", "help", "logging="]
    try:
        optlist,args = getopt.getopt(argv, 'x', argspec)
    except getopt.GetoptError:
        print 'bad option: use "--help" to get help'
        raise
    optdict = dict(optlist)

    def usage():
        print 'options:'
        print ' --db file.db              - file contains a serialized MatrixDB'
        print ' --db file1.cfacts1:...    - files are parsable with MatrixDB.loadFile()'
        print ' --prog file.ppr           - file is parsable as tensorlog rules'
        print ' --trainData file.exam     - optional: file is parsable with Dataset.loadExamples'
        print ' --trainData file.dset     - optional: file is a serialized Dataset'
        print ' --testData file.exam      - optional:'
        print ' --proppr                  - if present, assume the file has proppr features with'
        print '                             every rule: {ruleid}, or {all(F): p(X,...),q(...,F)}' 
        print ' --logging level'
        print ''
        print 'Notes: for --db, --trainData, and --testData, you are allowed to specify either a' 
        print 'serialized, cached object (like \'foo.db\') or a human-readable object that can be'
        print 'serialized (like \'foo.cfacts\'). In this case you can also write \'foo.db|foo.cfacts\''
        print 'and the appropriate uncache routine will be used.'

    if '--logging' in optdict:
        level = optdict['--logging']
        if level=='debug':
            logging.basicConfig(level=logging.DEBUG)
        elif level=='warn':
            logging.basicConfig(level=logging.WARN)
        elif level=='error':
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)

    if '--help' in optdict: 
        usage()
        exit(0)
    if (not '--db' in optdict) or (not '--prog' in optdict):
        usage()
        assert False,'--db and --prog are required options'

    db = parseDBSpec(optdict['--db'])
    optdict['--db'] = db
    optdict['--prog'] = parseProgSpec(optdict['--prog'],db,proppr=('--proppr' in optdict))

    for key in ('--trainData','--testData'):
        if key in optdict:
            optdict[key] = parseDatasetSpec(optdict[key],db)

    # let these be also indexed by 'train', 'prog', etc
    for key,val in optdict.items():
        optdict[key[2:]] = val

    return optdict,args

def isUncachefromSrc(s): return s.find("|")>=0
def getCacheSrcPair(s): return s.split("|")

def parseDatasetSpec(spec,db):
    """Parse a specification for a dataset, see usage() for parseCommandLine"""
    if isUncachefromSrc(spec):
        cache,src = getCacheSrcPair(spec)
        assert src.endswith(".examples") or src.endswith(".exam"), 'illegal --train or --test file'
        return dataset.Dataset.uncacheExamples(cache,db,src,proppr=src.endswith(".examples"))
    else:
        assert spec.endswith(".examples") or spec.endswith(".exam"), 'illegal --train or --test file'
        return dataset.Dataset.loadExamples(db,spec,proppr=spec.endswith(".examples"))

def parseDBSpec(spec):
    """Parse a specification for a database, see usage() for parseCommandLine"""
    if isUncachefromSrc(spec):
        cache,src = getCacheSrcPair(spec)
        return matrixdb.MatrixDB.uncache(cache,src)
    elif spec.endswith(".db"):
        return matrixdb.MatrixDB.deserialize(spec)
    elif spec.endswith(".cfacts"):
        return matrixdb.MatrixDB.loadFile(spec)
    else:
        assert False,'illegal --db spec %s' %spec
    
def parseProgSpec(spec,db,proppr=False):
    """Parse a specification for a Tensorlog program,, see usage() for parseCommandLine"""
    return ProPPRProgram.loadRules(spec,db) if proppr else Program.loadRules(spec,db)

#
# sample main: python tensorlog.py test/fam.cfacts 'rel(i,o)' 'rel(X,Y):-spouse(X,Y).' william
#

if __name__ == "__main__":
    
    print "Tensorlog v%s (C) William W. Cohen and Carnegie Mellon University, 2016" % VERSION

    optdict,args = parseCommandLine(sys.argv[1:])
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
        print "- The interpreter it is really only useful with the -i option."


