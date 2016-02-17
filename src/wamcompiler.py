# (C) William W. Cohen and Carnegie Mellon University, 2016

import collections
import sys

import parser

# Compile prolog rules to a variant of a Warren abstract machine.

##############################################################################
## A compiled program
##############################################################################

class Program(object):
    """Holds a modified Warran abstract machine program, consisting of:

    1)instructions = a list of tuples (opcode,arg1,...)

    2) labels = a defaultdict such that labels["p/n"] is a list of
     addresses (ie, indices in instructions) where the instructions
     for the clauses of p/n start.

    3) instLabels = a dict such that instLabels[i] is the label
    given to instruction i, if there is such a label.
    """
    def __init__(self):
        self.instructions = []
        self.labels = collections.defaultdict(list)
        self.instLabels = {}

    def append(self,instr):
        """Append a single instruction."""
        self.instructions.append(instr)

    def truncateTo(self,addr):
        """Remove all instructions past given address."""
        self.instructions = self.instructions[:addr]

    def insertLabel(self,key):
        addr = len(self.instructions)
        self.labels[key].append(addr)
        self.instLabels[addr] = key

    def ithInstructionAsString(self,i):
        inst = self.instructions[i]
        return '%-3d %8s  %s %s' % (i,self.instLabels.get(i,''),inst[0],inst[1:])

    def listing(self,lo=0):
        for i in range(lo,len(self.instructions)):
            print self.ithInstructionAsString(i)

    def sourceListing(self):
        for i in range(len(self.instructions)):
            if self.instructions[i][0]=='comment':
                print self.instructions[i][1]

##############################################################################
## The compiler
##############################################################################

class Compiler(object):
    """Compile prolog rules into a wam program."""

    def compileRules(self,rc,wamprog=None):
        if wamprog==None: wamprog = Program()
        for key in rc.index:
            for rule in rc.index[key]:
                rule.variabilize()
                wamprog.append(('comment',str(rule)))
                wamprog.insertLabel(key)
                self.compileRule(rule,wamprog)
        return wamprog

    def compileRule(self,rule,wamprog):
        # allocate variables
        nvars = rule.nvars
        if nvars>0:
            wamprog.append(('allocate',nvars,list(reversed(rule.variableList))))
        previousVars = set()
        # compile code to match the head
        if rule.lhs:
            for i,a in enumerate(rule.lhs.args):    
                relativeHeapIndex = -len(rule.lhs.args) + i
                if parser.isProcessedConstant(a): 
                    wamprog.append(('unifyconst',a,relativeHeapIndex))
                elif a in previousVars:
                    wamprog.append(('unifyboundvar',a,relativeHeapIndex))
                else:
                    wamprog.append(('initfreevar',a,relativeHeapIndex))
                    previousVars.add(a)
        # compile the body
        for g in rule.rhs:
            self.compileGoal(g,wamprog,previousVars)
        wamprog.append(('returnp',))

    def compileGoal(self,g,wamprog,previousVars):
        for i,a in enumerate(g.args):
            if parser.isProcessedConstant(a): 
                wamprog.append(('pushconst',a))
            elif a in previousVars:
                wamprog.append(('pushboundvar',a))
            else:
                wamprog.append(('pushfreevar',a))
                previousVars.add(a)
        wamprog.append(('callp','%s/%s' % (g.functor,g.arity)))



##############################################################################
## a test driver
##############################################################################

if __name__ == "__main__":

    if True:
        rules = parser.RuleCollection()
        for ruleString in sys.argv[1:]:
            #print ruleString,"==>",featureBlockNT.parseString(ruleString)
            ptree = parser.ruleNT.parseString(ruleString)
            print ruleString,"==>",ptree
            if 'lhs' in ptree: print 'lhs:',ptree['lhs']
            if 'rhs' in ptree: print 'rhs',ptree['rhs']
            if 'features' in ptree: print 'features',ptree['features']
            r = parser.Parser._convertRule(ptree)
            print 'rule:',r
            print 'rule features',r.features,'map-str',map(str,r.features)
            print 'rule findall',r.findall,
            if r.findall: print 'map-str',map(str,r.findall)
            else: print
            r.variabilize()
            print 'variabilized rule:',r

            rules.add(r)
        rules.listing()
        wp = Compiler().compileRules(rules)
        wp.listing()

    else:
        rules = Parser.parseFile(sys.argv[1])
        for f in sys.argv[2:]:
            rules = Parser.parseFile(f,rules)

        rules.listing()
        wp = Compiler().compileRules(rules)
        wp.listing()
        
