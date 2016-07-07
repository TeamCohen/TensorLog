import sys
import re

if __name__=="__main__":
    #usage: python convert-rules.py train.cfacts input-recursive-rules.ppr output-rules ruleids
    ruleFP = open(sys.argv[3], 'w')
    idFactFP = open(sys.argv[4], 'w')
    predicates = set()
    for line in open(sys.argv[2]):
        m = re.match('.*\{(\w+)\}.*',line)
        assert m,'bad line: '+line.strip()
        idFactFP.write('ruleid\t%s' % m.group(1) + '\n')
        ruleFP.write('i_'+line)
        
    #scan to find predicates
    for line in open(sys.argv[1]):
        (p,x,y) = line.strip().split("\t")
        predicates.add(p)
    for p in predicates:
        ruleFP.write('i_%s(X,Y) :- %s(X,Y) {base_%s}.\n' % (p,p,p))
        idFactFP.write('ruleid\tbase_%s\n' % p)

