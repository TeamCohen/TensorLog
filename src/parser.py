# (C) William W. Cohen and Carnegie Mellon University, 2016

import collections

import symtab as syt

#
# Parse prolog rules in one of these sample formats
#
# p(X,Y) :- q(X,Z), r(Z,X).         # normal prolog clause
# p(X,Y,Z) :- .                     # unit clause
# p(X,Y) :- q(X,Z) {f(Y,X)}.        # normal prolog clause plus a 'feature'  
# p(X,Y) :- q(X,Z) {f(Y,X),g(Y)}.   # multiple 'features'  
# p(X,Y) :- q(X,Z) {f(W) : g(Y,W)}. # features geberated by a 'findall'
#                                   #  ie for all solutions of g(Y,W), 
#                                   #  produce a feature f(W)
#
# TODO: remove the stuff that's not supported in TensorLog

##############################################################################
## data structures to encode rules
##############################################################################

def isProcessedConstant(a):
    return not isProcessedVariable(a)

def isProcessedVariable(a):
    return type(a)==type(0)

def isVariableAtom(a):
    return a[0].isupper() or a[0]=='_' 

class Goal(object):
    """A prolog goal, eg brotherOf(X,Y)."""
    def __init__(self,functor,args):
        self.functor = functor
        self._setArgs(args)
        
    def _setArgs(self,args):
        self.args = args
        self.arity = len(args)

    def __str__(self):
        if self.arity: return "%s(%s)" % (self.functor,",".join(map(str,self.args)))
        else: return self.functor

class Rule(object):
    """A prolog rule.  The lhs is a goal, the rhs a list of goals, so the
    rule's format is "lhs :- rhs."  The features for a rule are, in
    general, of the form "features : findall", where 'findall' and
    'features' are lists of goals.  Features are produced as follows:
    after binding the head of the rule, you find all solutions to the
    'findall' part (the "generator"), and for each solution, create a
    feature corresponding to a bound version of each goal g in
    'features'.
    """

    def __init__(self,lhs,rhs,features=None,findall=None):
        self.lhs = lhs
        self.rhs = rhs
        self.features = features
        self.findall = findall
        self.variableList = None
        self.nvars = -1

    def variabilize(self):
        """To simplify compilation - convert the variables to integer indices,
        -1,-2, ... and save their original names in "variableList",
        and the number of distinct variables in 'nvars."""
        if self.nvars>=0:
            pass #already done
        else:
            varTab = syt.SymbolTable()
            def convertArgs(args):
                return map(lambda a: -varTab.getId(a) if isVariableAtom(a) else a, args)
            def convertGoal(g):
                return Goal(g.functor, convertArgs(g.args))
            if self.lhs: self.lhs = convertGoal(self.lhs)
            self.rhs = map(convertGoal, self.rhs)
            if self.features:
                self.features = map(convertGoal, self.features)
            if self.findall:
                self.findall = map(convertGoal, self.findall)                
            self.variableList = varTab.getSymbolList()
            self.nvars = len(self.variableList)

    def __str__(self):
        vars = "  #v:"+str(self.variableList) if self.variableList else ''
        findalls = ' : '+",".join(map(str,self.findall)) if self.findall else ''
        features = ' {' + ",".join(map(str,self.features)) + findalls + '}' if self.features else ''
        return str(self.lhs) + " :- " + ", ".join(map(str,self.rhs)) + features + vars + '.'

class RuleCollection(object):
    """A set of prolog rules, indexed by functor and arity."""
    
    def __init__(self):
        self.index = collections.defaultdict(list)
    
    def _key(self,g):
        return '%s/%d' % (g.functor,g.arity) 

    def add(self,r):
        key = self._key(r.lhs)
        self.index[key] += [r]

    def size(self):
        return sum(len(self.index[k]) for k in self.index.keys())

    def rulesFor(self,g):
        return self.index[self._key(g)]

    def mapRules(self,mapfun):
        for key in self.index:
            self.index[key] = map(mapfun, self.index[key]) 

    def listing(self):
        for key in self.index:
            print'% rules for',key
            for r in self.index[key]:
                print r

##############################################################################
## the parser
##############################################################################

from pyparsing import Word, CharsNotIn, alphas, alphanums, delimitedList, nestedExpr, Optional, Group, QuotedString

atomNT = Word( alphanums+"_" ) |  QuotedString(quoteChar="'",escChar="\\")
goalNT = atomNT + Optional("(" + delimitedList(atomNT) + ")")
goalListNT = Optional(delimitedList(Group(goalNT)))
featureFindAllNT = Optional(":" + delimitedList(Group(goalNT)))
featureTemplateNT = delimitedList(Group(goalNT))
featureBlockNT = Optional("{" + featureTemplateNT('ftemplate') + featureFindAllNT('ffindall') + "}")
ruleNT = goalNT("lhs") + ":-" + goalListNT("rhs") +  featureBlockNT("features") + "."

class Parser(object):

    @staticmethod
    def _convertGoal(ptree):
        return Goal(ptree[0], ptree[2:-1])

    @staticmethod
    def _convertRule(ptree):
        if 'rhs' in ptree: 
            tmpRhs = map(Parser._convertGoal, ptree['rhs'].asList())
        else: 
            tmpRhs = []
        if not 'features' in ptree:
            return Rule(Parser._convertGoal(ptree['lhs']),tmpRhs,None,None) 
        else:
            if not 'ffindall' in ptree:
                featureList = ptree['ftemplate'].asList()
                tmpFeatures = map(Parser._convertGoal, featureList)
                return Rule(Parser._convertGoal(ptree['lhs']),tmpRhs,tmpFeatures,None) 
            else:
                featureList = ptree['ftemplate'].asList()
                tmpFeatures = map(Parser._convertGoal, featureList)
                findallList = ptree['ffindall'].asList()[1:]
                tmpFindall = map(Parser._convertGoal, findallList)                
                return Rule(Parser._convertGoal(ptree['lhs']),tmpRhs,tmpFeatures,tmpFindall) 

    @staticmethod
    def parseGoal(s):
        """Convert a string to a goal."""
        return Parser._convertGoal(goalNT.parseString(s))
	
    @staticmethod
    def parseGoalList(s):
        """Convert a string to a goal list."""
        return map(Parser._convertGoal, goalListNT.parseString(s).asList())
		
    @staticmethod
    def parseRule(s):
        """Convert a string to a rule."""
        return Parser._convertRule(ruleNT.parseString(s))

    @staticmethod
    def parseQuery(s):
        """Convert a string to a headless rule (no lhs)"""
        result = Parser.parseRule('dummy :- %s\n' % s)
        result.lhs = None
        return result

    @staticmethod
    def parseFile(file,rules = None):
        """Extract a series of rules from a file."""
        if not rules: rules = RuleCollection()
        buf = ""
        for line in open(file,'r'):
            if not line[0]=='#':
                buf += line
        try:
            for (ptree,lo,hi) in ruleNT.scanString(buf):
                rules.add(Parser._convertRule(ptree))
            return rules
        except KeyError:
            print 'error near ',lo,'in',file
        return rules

