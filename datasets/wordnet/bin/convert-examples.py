import sys
import tensorlog
import re

def cvtExamples(fIn,fOut,prefix,targetPred):
    fp = open(fOut,'w')
    regex = re.compile('interp\((i_\w+),(\w+),(\w+)')
    for line in open(fIn):
        parts = line.strip().split("\t")
        m = regex.search(parts[0])
        pred = m.group(1)
        queryX = m.group(2)
        pos = []
        if pred==targetPred:
            for ans in parts[1:]:
                #print pred,queryX,line.strip()
                if ans[0]=='+':
                    m = regex.search(ans[1:])
                    pos.append(m.group(3))
                #print pred,queryX,pos,line.strip()
            if pos:
                for p in pos: fp.write('%s_%s\t%s\t%s\n' % (prefix,pred,queryX,p))
    print 'produced',fOut

if __name__ == "__main__":
    for rel in ['hyponym','derivationally_related_form','member_meronym','member_holonym']:
        for pref in ['train','valid']:
            cvtExamples('raw/%s.examples' % pref, 'wnet-%s-%s.cfacts' % (rel,pref), pref, 'i_%s' % rel)


