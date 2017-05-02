# (C) William W. Cohen and Carnegie Mellon University, 2016

# mode declarations for Tensorlog (and eventually type declarations)

from tensorlog import parser

def asMode(spec):
    """Convert strings like "foo(i,o)" or "foo/io" to ModeDeclarations.
    Or, if given a ModeDeclaration object, return that object.
    """
    if type(spec)==type("") and spec.find("/")>=0:
        functor,rest = spec.split("/")
        return ModeDeclaration(parser.Goal(functor,list(rest)))
    elif type(spec)==type(""):
        return ModeDeclaration(spec)
    else:
        return spec

class AbstractDeclaration(object):
    """Mode - or later - type - declaration for a predicate."""
    def __init__(self,goal):
        if type(goal)==type(""):
            goal = parser.Parser.parseGoal(goal)
        self.prototype = goal
        self._key = str(goal)
    def args(self):
        return self.prototype.args
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
    def __init__(self,goal,strict=True):
        super(ModeDeclaration,self).__init__(goal)
        if strict:
            for a in self.prototype.args:
                assert a=='i' or a=='o','arguments to a ModeDeclaration should be "i" or "o" (not %s for mode %r)' % (a,self.prototype)
    def isInput(self,i):
        return self.arg(i)=='i'
    def isOutput(self,i):
        return self.arg(i)=='o'
    def isConst(self,i):
        return not self.isInput(i) and not self.isOutput(i)
    def __str__(self):
        return self.functor + "/" + "".join(self.prototype.args)

class TypeDeclaration(AbstractDeclaration):
    """Declare allowed types for a goal, eg hasWord(doc,word).
    """
    def __init__(self,goal):
        super(TypeDeclaration,self).__init__(goal)
    def getType(self,i):
        return self.arg(i)
    def typeSet(self):
        return set(self.prototype.args)
