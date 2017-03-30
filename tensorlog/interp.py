# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# top-level constructs for Tensorlog - Programs and Interpreters
#

import sys
import logging
import collections

from tensorlog import comline
from tensorlog import dataset
from tensorlog import declare
from tensorlog import parser
from tensorlog import program

# interactive debugger is sort of optional, and it depends on a bunch
# of stuff that might not be present, so don't require this import
try:
  from tensorlog import debug
  DEBUGGER_AVAILABLE = True
except ImportError:
  logging.warn('debug module not imported')
  DEBUGGER_AVAILABLE = False

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
        print "ti.eval(\"functor/mode\",\"c\",inputType=type1,outputType=type2): evaluate a function on a database constant c"
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

    def eval(self,modeSpec,sym,inputType=None,outputType=None):
        mode = declare.asMode(modeSpec)
        fun = self.prog.getFunction(mode)
        outputType = outputType or fun.outputType
        inputType = inputType or fun.inputTypes[0]
        tmp = self.prog.evalSymbols(mode,[sym],typeName=inputType)
        result = self.prog.db.rowAsSymbolDict(tmp,typeName=outputType)
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
      assert self.db.isTypeless(),'cannot debug a db with declared types'
      X = self.db.onehot(sym)
      dset = dataset.Dataset({mode:X},{mode:self.db.zeros()})
      debug.Debugger(self.prog,mode,dset,gradient=False).mainloop()

    def debugDset(self,modeSpec,test=False):
      if not DEBUGGER_AVAILABLE:
        logging.warn('debugger is not available in this environment')
        return
      assert self.db.isTypeless(),'cannot debug a db with declared types'
      fullDataset = self.testData if test else self.trainData
      if fullDataset==None:
        print 'train/test dataset is not specified on command line?'
      else:
        mode = declare.asMode(modeSpec)
        dset = fullDataset.extractMode(mode)
        debug.Debugger(self.prog,mode,dset,gradient=True).mainloop()

#
# sample main
#

if __name__ == "__main__":

    print "Tensorlog v%s (C) William W. Cohen and Carnegie Mellon University, 2016" % program.VERSION

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
