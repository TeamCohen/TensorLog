import sys
import tensorlog
import re

def cvtRules(fIn,fOut,rIdOut):
    rn = 0
    fp = open(fOut,'w')
    fp2 = open(rIdOut,'w')
    regex = re.compile('^(\w+)\((\w+),(.*)')
    def fixLit(lit):
        m = regex.match(lit)        
        return '%s(%s' % (m.group(2),m.group(3))

    for line in open(fIn):
        rn += 1
        if not line.startswith("#") and not line.startswith("interp(P") and not line.startswith("learnedPred(P") and line.strip():
            head,bodyFeat = line.strip().split(" :- ")
            body,feat0 = bodyFeat.split(" {")
            bodyLits = body.split(", ")
            fp.write(fixLit(head))
            fp.write(' :- ')
            fp.write(", ".join(map(fixLit,bodyLits)))
            fp.write(' {r%d}.\n' % rn)
            fp2.write('rule\tr%d\n' % rn)

if __name__ == "__main__":
    cvtRules('raw/train-learned.ppr','wnet-learned.ppr', 'wnet-ruleids.cfacts')


