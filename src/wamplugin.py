# (C) William W. Cohen and Carnegie Mellon University, 2016

import prover
import collections
import util

import traceback

##############################################################################
# Plugins - extensions to the WAM
##############################################################################

# Builtin-functions that use 'provers' to implement a command-line interpreter

def pluginCommandLineInterfaceFunction0(wamInterp):
    """Arity-zero version of cli."""
    runCommandLineInterface(wamInterp)

def pluginCommandLineInterfaceFunction1(wamInterp):
    """Arity-one version of cli, eg cli(dfs)."""
    proverName = wamInterp.getConstantArg(1,1)
    runCommandLineInterface(wamInterp,proverName)
    
def pluginCommandLineInterfaceFunction2(wamInterp):
    """Arity-two version of cli, eg cli(dfs,'maxDepth:4,...')."""
    proverName = wamInterp.getConstantArg(2,1)
    maxDepth = int(wamInterp.getConstantArg(2,2))
    runCommandLineInterface(wamInterp,proverName,maxDepth)

#TODO: this is sort of functional but buggy
def runCommandLineInterface(wamInterp,proverName='dfs',maxDepth=10):
    
    print 'running command line interface with prover',proverName,'maxDepth',maxDepth
    def doCommand(proverName,com):
        print 'executing "%s"' % com
        if proverName=='dfs':
            for s in wamInterp.solutions(queryString=com):
                print s
        elif proverName=='trace':
            wamInterp.trace(com)

    while (True):
        try:
            com = raw_input("ProPPR %s> " % proverName)
            if com=='exit.':
                return
            doCommand(proverName,com)
        except EOFError:
            break
        except Exception:
            print 'error: executing "%s"' % com
            traceback.print_exc()


class WamPlugin(object):
    """Abstract extension to a WAM program."""
    def claim(self,jumpTo):
        """Return True if this plugin should be called to implement this predicate/arity pair."""
        assert False,'abstract method called'
    def outlinks(self,state,wamInterp):
        """Yield a list of successor states, not including the restart state."""
        assert False,'abstract method called'
    def implementsDegree(self):
        """True if the subclass implements a degree() function that's quicker than computing the outlinks."""
        return False
    def degree(self,jumpTo,state,wamInterp):
        """Return the number of outlinks, or else throw an error if implementsDegree is false."""
        assert False,'degree method not implemented'
        
class PlugInCollection(WamPlugin):
    """Used for collections of simple built-in plugins."""
    def __init__(self):
        self.registery = {}
        self.helpText = {}
        self.fd = {'builtin':1}
    def register(self,jumpTo,fun,help='no help available'):
        self.registery[jumpTo] = fun
        self.helpText[jumpTo] = help
    def claim(self,jumpTo):
        return (jumpTo in self.registery)
    def outlinks(self,state,wamInterp):
        assert False,'abstract method called'

class FilterPluginCollection(PlugInCollection):
    """Used for built-ins that may or may not succeed, like 'neq'."""
    def __init__(self):
        PlugInCollection.__init__(self)
    def implementsDegree(self,state):
        return False
    def outlinks(self,state,wamInterp):
        jumpTo = state.jumpTo
        filterFun = self.registery[jumpTo]
        if not filterFun(wamInterp):
            wamInterp.restoreState(state)
            wamInterp.returnp()
            wamInterp.state.failed = True
        else:
            wamInterp.restoreState(state)
            wamInterp.returnp()
            wamInterp.executeWithoutBranching()
        yield wamInterp.saveState()

class SideEffectPluginCollection(PlugInCollection):
    """Used for built-ins that always succeed, but have some side effect, like 'listing'."""
    def __init__(self):
        PlugInCollection.__init__(self)
    def outlinks(self,state,wamInterp):
        jumpTo = state.jumpTo
        #execute the side effect, eg, 'listing'
        sideEffectFun = self.registery[jumpTo]
        sideEffectFun(wamInterp)
        #return from the subroutine
        wamInterp.restoreState(state)
        wamInterp.returnp()
        wamInterp.executeWithoutBranching()
        yield wamInterp.saveState()
    def implementsDegree(self,state):
        return True
    def degree(self,state,wamInterp):
        return 1

def builtInPlugins():
    """Create a list of pre-defined WAM extensions."""
    result = []
    filtPlugin = FilterPluginCollection()
    def neqFun(wamInterp):
        arg1 = wamInterp.getConstantArg(2,1)
        arg2 = wamInterp.getConstantArg(2,2)
        assert arg1!=None and arg2!=None,'cannot call neq/2 unless both variables are bound'
        #print 'testing',arg1,'neq',arg2,'result',(arg1!=arg2)
        return arg1!=arg2
    filtPlugin.register(
        'neq/2',neqFun,'inequality test - note, this must be applied to bound variables')
    result.append(filtPlugin)

    sePlugin = SideEffectPluginCollection()
    def wp_listingFun(wamInterp):
        wamInterp.wamprog.listing()
    sePlugin.register(
        'wp_listing/0',wp_listingFun,'list the compiled wam program')
    def listingFun(wamInterp):
        wamInterp.wamprog.sourceListing()
    sePlugin.register(
        'listing/0',listingFun,'list the ProPPR source program')
    def helpFun(wamInterp):
        for functorArity,help in sePlugin.helpText.items():
            print functorArity,'\t',help
        for functorArity,help in filtPlugin.helpText.items():
            print functorArity,'\t',help
    sePlugin.register(
        'help/0',helpFun,'this help message')

    #import plugin functions from prover that implement a command-line interface
    sePlugin.register(
        'cli/0',pluginCommandLineInterfaceFunction0,'launch a command line interface')
    sePlugin.register(
        'cli/1',pluginCommandLineInterfaceFunction1,'launch a command line interface')
    sePlugin.register(
        'cli/2',pluginCommandLineInterfaceFunction2,'launch a command line interface')
    result.append(sePlugin)

    result.append(sePlugin)
    return result

##############################################################################



class FactPlugin(WamPlugin):
    """A 'extensional database' of facts."""

    # used to create and view a simple EDB.

    def __init__(self,label='graphEDB'):

        self._index = collections.defaultdict(lambda:collections.defaultdict(list))
        self._arity = {}
        self._label = label
        
    def addFact(self,f,src,dst):
        """An an arc to a graph-based EDB."""
        n = len(dst)+1
        key = "%s/%d" % (f,n)
        self._index[key][src].append(dst)
        self._arity[key] = n

    def claim(self, jumpTo):
        return jumpTo in self._index

    def degree(self,jumpTo,state,wamInterp):
        srcConst = wamInterp.getConstantArg(2,1)
        assert srcConst,'predicate '+jumpTo+' called with non-constant first argument!'
        return len(self._index[jumpTo][srcConst])

    def outlinks(self,state,wamInterp):
        n = self._arity[state.jumpTo]
        srcConst = wamInterp.getConstantArg(n,1)
        assert srcConst,'predicate '+state.jumpTo+' called with non-constant first argument!'
        dstConsts = []
        for i in range(1,n):
            #print 'getting arg',i,'arity',n,'index',i+1
            dstConsts.append(wamInterp.getConstantArg(n,i+1))
        for s in self.outlinksPerSource(state,wamInterp,srcConst,dstConsts):
            yield s
    
    def outlinksPerSource(self,state,wamInterp,srcConst,dstConsts):
        n = self._arity[state.jumpTo]
        values = self._index[state.jumpTo][srcConst]
        if values:
            w = 1.0/len(values)
            valListMatches = True
            for valList in values:
                wamInterp.restoreState(state)
                for j in range(len(valList)):
                    if not dstConsts[j]:
                        wamInterp.setArg(n,j+2,valList[j])
                    elif dstConsts[j]!=valList[j]:
                        valListMatches = False
                if valListMatches:
                    wamInterp.returnp()
                    wamInterp.executeWithoutBranching()
                    yield wamInterp.saveState()

    @staticmethod
    def load(file):
        """Return a simpleGraphComponent with all the components loaded from
        a file.  The format of the file is that each line is a tab-separated 
        triple of edgelabel, sourceNode, destNode."""
        p = FactPlugin(label=file)
        for line in util.linesOf(file,interval=100000):
            if not line.startswith("#") and line.strip():
                try:
                    parts = line.strip().split("\t")
                    functor = parts[0].strip()
                    src = parts[1].strip()
                    dsts = map(lambda x:x.strip(), parts[2:])
                    p.addFact(functor,src,dsts)
                except KeyError:
                    print 'bad line:',line
        return p

class GraphPlugin(WamPlugin):
    """A 'extensional database' - restricted to be a labeled directed
    graph, or equivalently, a set of f(+X,-Y) unit predicates."""

    # used to create and view a graph-based EDB.

    def __init__(self,label='graphEDB'):
        self._index = collections.defaultdict(lambda:collections.defaultdict(list))
        self._label = label
        
    def addEdge(self,f,src,dst):
        """An an arc to a graph-based EDB."""
        self._index[f+"/2"][src] += [dst]

    def claim(self, jumpTo):
        return jumpTo in self._index

    def degree(self,jumpTo,state,wamInterp):
        srcConst = wamInterp.getConstantArg(2,1)
        assert srcConst,'predicate '+jumpTo+' called with non-constant first argument!'
        return len(self._index[jumpTo][srcConst])

    def outlinks(self,state,wamInterp):
        srcConst = wamInterp.getConstantArg(2,1)
        dstConst = wamInterp.getConstantArg(2,2)
        if srcConst:
            for s in self.outlinksPerSource(state,wamInterp,srcConst,dstConst):
                yield s
        else:
            for src in self._index[state.jumpTo].keys():
                wamInterp.restoreState(state)
                wamInterp.setArg(2,1,src)
                srcState = wamInterp.saveState()
                for s in self.outlinksPerSource(srcState,wamInterp,src,dstConst):
                    yield s
    
    def outlinksPerSource(self,state,wamInterp,srcConst,dstConst):
        values = self._index[state.jumpTo][srcConst]
        if values:
            w = 1.0/len(values)
            for val in values:
                if dstConst and val==dstConst:
                    wamInterp.restoreState(state)
                    wamInterp.returnp()
                    wamInterp.executeWithoutBranching()
                    yield wamInterp.saveState()
                elif not dstConst:
                    wamInterp.restoreState(state)
                    wamInterp.setArg(2,2,val)
                    wamInterp.returnp()
                    wamInterp.executeWithoutBranching()
                    yield wamInterp.saveState()
    @staticmethod
    def load(file):
        """Return a simpleGraphComponent with all the components loaded from
        a file.  The format of the file is that each line is a tab-separated 
        triple of edgelabel, sourceNode, destNode."""
        p = GraphPlugin(label=file)
        for line in util.linesOf(file,interval=100000):
            if not line.startswith("#") and line.strip():
                try:
                    edgeLabel,src,dst = line.strip().split("\t")
                    edgeLabel = edgeLabel.strip()
                    src = src.strip()
                    dst = dst.strip()
                    p.addEdge(edgeLabel,src,dst)
                except KeyError:
                    print 'bad line:',line
        return p
