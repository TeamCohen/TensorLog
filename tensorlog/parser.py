# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import collections
import logging

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

from tensorlog import config

conf = config.Config()
conf.syntax = 'proppr';        conf.help.syntax = "Should be 'pythonic' or 'proppr'"

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

    def __repr__(self):
        return 'Goal(%r,%r)' % (self.functor,self.args)


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
      return self.asString()

    def asString(self,syntax=None):
      if syntax is None: syntax=conf.syntax
      vars = "  #v:"+str(self.variableList) if self.variableList else ''
      if syntax == 'proppr':
        findalls = ' : '+",".join(map(str,self.findall)) if self.findall else ''
        features = ' {' + ",".join(map(str,self.features)) + findalls + '}' if self.features else ''
        return str(self.lhs) + " :- " + ", ".join(map(str,self.rhs)) + features + vars + '.'
      else:
        findalls = ' | '+" & ".join(map(str,self.findall)) if self.findall else ''
        features = ' // ' + " & ".join(map(str,self.features)) + findalls if self.features else ''
        return str(self.lhs) + " <= " + " & ".join(map(str,self.rhs)) + features + vars


class RuleCollection(object):
    """A set of prolog rules, indexed by functor and arity."""

    def __init__(self,syntax=None):
        self.index = collections.defaultdict(list)
        self.syntax = syntax or conf.syntax

    def _key(self,g):
        return '%s/%d' % (g.functor,g.arity)

    def add(self,r):
        key = self._key(r.lhs)
        self.index[key] += [r]

    def size(self):
        return sum(len(self.index[k]) for k in self.index.keys())

    def rulesFor(self,g):
        return self.index.get(self._key(g))

    def mapRules(self,mapfun):
        for key in self.index:
            self.index[key] = map(mapfun, self.index[key])

    def listing(self):
        for key in self.index:
            print'% rules for',key
            for r in self.index[key]:
                print r.asString(syntax=self.syntax)

    def __iter__(self):
        for key in self.index:
            for r in self.index[key]:
                yield r

##############################################################################
## the parser
##############################################################################

from pyparsing import Word, CharsNotIn, alphas, alphanums, delimitedList, nestedExpr, Optional, Group, QuotedString

class Parser(object):

  def __init__(self,syntax=None):
    self.setSyntax(syntax or conf.syntax)

  def setSyntax(self,syntax):
    self.syntax = syntax
    self.atomNT = Word( alphanums+"_$" ) |  QuotedString(quoteChar="'",escChar="\\")
    self.goalNT = self.atomNT + Optional("(" + delimitedList(self.atomNT) + ")")
    if self.syntax=='proppr':
      self.goalListNT = Optional(delimitedList(Group(self.goalNT)))
      self.featureFindAllNT = Optional(":" + delimitedList(Group(self.goalNT)))
      self.featureTemplateNT = delimitedList(Group(self.goalNT))
      self.featureBlockNT = Optional("{" + self.featureTemplateNT('ftemplate') + self.featureFindAllNT('ffindall') + "}")
      self.ruleNT = self.goalNT("lhs") + ":-" + self.goalListNT("rhs") +  self.featureBlockNT("features") + "."
    else:
      self.goalListNT = Optional(delimitedList(Group(self.goalNT), delim="&"))
      self.featureFindAllNT = Optional("|" + delimitedList(Group(self.goalNT), delim="&"))
      self.featureTemplateNT = delimitedList(Group(self.goalNT), delim="&")
      self.featureBlockNT = Optional("//" + self.featureTemplateNT('ftemplate') + self.featureFindAllNT('ffindall'))
      self.ruleNT = self.goalNT("lhs") + "<=" + self.goalListNT("rhs") +  self.featureBlockNT("features")

  def _convertGoal(self,ptree):
    return Goal(ptree[0], ptree[2:-1])

  def _convertRule(self,ptree):
    if 'rhs' in ptree:
      tmpRhs = map(self._convertGoal, ptree['rhs'].asList())
    else:
      tmpRhs = []
    if not 'features' in ptree:
      return Rule(self._convertGoal(ptree['lhs']),tmpRhs,None,None)
    else:
      if not 'ffindall' in ptree:
        featureList = ptree['ftemplate'].asList()
        tmpFeatures = map(self._convertGoal, featureList)
        return Rule(self._convertGoal(ptree['lhs']),tmpRhs,tmpFeatures,None)
      else:
        featureList = ptree['ftemplate'].asList()
        tmpFeatures = map(self._convertGoal, featureList)
        findallList = ptree['ffindall'].asList()[1:]
        tmpFindall = map(self._convertGoal, findallList)
        return Rule(self._convertGoal(ptree['lhs']),tmpRhs,tmpFeatures,tmpFindall)

  def parseGoal(self,s):
    """Convert a string to a goal."""
    return self._convertGoal(self.goalNT.parseString(s))

  def parseGoalList(self,s):
    """Convert a string to a goal list."""
    return map(self._convertGoal, self.goalListNT.parseString(s).asList())

  def parseRule(self,s):
    """Convert a string to a rule."""
    return self._convertRule(self.ruleNT.parseString(s))

  def parseQuery(self,s):
    """Convert a string to a headless rule (no lhs)"""
    result = Parser().parseRule('dummy :- %s\n' % s)
    result.lhs = None
    return result

  def parseFile(self,filename,rules = None):
    """Extract a series of rules from a file."""
    if filename.endswith("tlog"): self.setSyntax('pythonic')
    if not rules: rules = RuleCollection(syntax=self.syntax)
    linebuf = []
    for line in open(filename):
      if not line[0]=='#':
        linebuf.append(line)
    buf = "".join(linebuf)
    try:
      first_time = True
      for (ptree,lo,hi) in self.ruleNT.scanString(buf):
        rules.add(self._convertRule(ptree))
        if first_time:
          unread_text = buf[:lo].strip()
          if len(unread_text)>0:
            logging.error('unparsed text at start of %s: "%s..."' % (filename,unread_text))
          first_time = False
      unread_text = buf[hi:].strip() if rules.size()>0 else buf
      if len(unread_text)>0:
        logging.error('unparsed text at end of %s: "...%s"' % (filename,unread_text))
      return rules
    except KeyError:
      print 'error near ',lo,'in',filename
      return rules

if __name__ == "__main__":
  p = Parser(syntax='pythonic')

  for f in sys.argv[1:]:
    print '\nparsed from file %r:' % f
    Parser().parseFile(f).listing()
