import sys
import tensorlog
import re

def cvtExamples(fIn,fOut):
    fp = open(fOut,'w')
    regex = re.compile('interp\((i_\w+),(\w+),(\w+)')
    for line in open(fIn):
        parts = line.strip().split("\t")
        m = regex.search(parts[0])
        pred = m.group(1)
        queryX = m.group(2)
        pos = []
        for ans in parts[1:]:
            #print pred,queryX,line.strip()
            if ans[0]=='+':
                m = regex.search(ans[1:])
                pos.append('s'+m.group(3))
            #print pred,queryX,pos,line.strip()
        if pos:
            fp.write('\t'.join([pred,'s'+queryX]+pos) + '\n')
    print 'produced',fOut

if __name__ == "__main__":
        for pref in ['train','valid']:
            cvtExamples('raw/%s.examples' % pref, 'inputs/wnet-%s.exam' % pref)


#    for rel in ['hypernym','hyponym','derivationally_related_form','member_meronym','member_holonym']:
#        for pref in ['train','valid']:
#            cvtExamples('raw/%s.examples' % pref, 'inputs/wnet-%s-%s.cfacts' % (rel,pref), pref, 'i_%s' % rel)


