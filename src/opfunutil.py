# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# utility classes used by funs.py or ops.ps
#


class OperatorOrFunction(object):

    #needed for visualizations

    def pprint(self,depth=0):
        """Return a list of strings that can be joined to form a textual view
        of self.  This should include all the substructures.
        """
        assert False, 'abstract method called'

    def pprintSummary(self):
        """A short summary string used in pprint() describing self.
        """
        assert False, 'abstract method called'

    def pprintComment(self):
        """A comment/provenance info for self.."""
        assert False, 'abstract method called'

    def children(self):
        """List of substructures."""
        assert False, 'abstract method called'
    
    def install(self,nextId=1):
        """Traverse all substructures and assign numeric ids to then."""
        assert False, 'abstract method called'


class MutableObject(object):
    """An object that one can attach properties to,
    to stick into a Scratchpad.
    """
    pass

class Scratchpad(object):
    """ Space for data, like function outputs and gradients, generated
    during eval and backprop. Typically a Scratchpad 'pad' will be
    indexed by the numeric id of an OperatorOrFunction object,
    eg "pad[id].output = foo" or "pad[id].delta = bar".
    """
    def __init__(self):
        self.d = dict()
    #override pad[id] to access d
    def __getitem__(self,key):
        if key not in self.d:
            self.d[key] = MutableObject()
        return self.d[key]
    def __setitem__(self,key,val):
        if key not in self.d:
            self.d[key] = MutableObject()
        self.d[key] = val

class Envir(object):
    """Holds a MatrixDB object, and a group of variable bindings for the
    variables used in message-passing.  The value to which variable 'foo'
    is bound is stored in env.register[foo], which is also written
    env[foo].  The backprop-ed delta is stored in env.delta[foo].
    """
    def __init__(self,db):
        self.register = {}
        self.delta = {}
        self.db = db
    def bindList(self,vars,vals):
        """Bind each variable in a list to the corresponding value."""
        assert len(vars)==len(vals)
        for i in range(len(vars)):
            self[vars[i]] = vals[i]
    def __repr__(self):
        return 'Envir(%r)' % self.register
    #override env[var] to access 'register'
    def __getitem__(self,key):
        return self.register[key]
    def __setitem__(self,key,val):
        self.register[key] = val

