import sys
import tensorlog
import re

ALLRELS = ['also_see', 'derivationally_related_form', 'has_part',
           'hypernym', 'hyponym', 'instance_hypernym', 'instance_hyponym',
           'member_holonym', 'member_meronym', 'member_of_domain_region',
           'member_of_domain_topic', 'member_of_domain_usage', 'part_of',
           'similar_to', 'synset_domain_region_of', 'synset_domain_topic_of',
           'synset_domain_usage_of', 'verb_group']

def cvtRules(fIn,fOut,rIdOut):
    rn = 0
    fp = open(fOut,'w')
    fp2 = open(rIdOut,'w')
    regex = re.compile('^(\w+)\((\w+),(.*)')
    def fixLit(lit):
        m = regex.match(lit)        
        return '%s(%s' % (m.group(2),m.group(3))

    for rel in ALLRELS:
        fp.write('i_%s(X,Y) :- %s(X,Y) {i_%s}.\n' % (rel,rel,rel)) 
        fp2.write('rule\ti_%s\n' % rel)

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
    cvtRules('raw/train-learned.ppr','inputs/wnet-learned.ppr', 'inputs/wnet-ruleids.cfacts')


