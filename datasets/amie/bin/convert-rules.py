# read an AMIE rules file, output a TensorLog rules file.
# /remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv
# sample input:
"""
Rule    Support Head coverage   Std. Confidence PCA Confidence  Positive Examples       Body size       Functional variable
?a  <isKnownFor>  ?b   => ?a  <created>  ?b     0.00017148      0.0008261       3.01%   100.00% 8       266     ?b
?a  <created>  ?b   => ?a  <isKnownFor>  ?b     0.00017148      0.03007519      0.08%   100.00% 8       9684    ?b
?a  <isLeaderOf>  ?b  ?a  <wasBornIn>  ?b   => ?a  <livesIn>  ?b        4.287E-05       0.00422833      40.00%  100.00% 2       5       ?a
?a  <wasBornIn>  ?b  ?a  <diedIn>  ?b   => ?a  <livesIn>  ?b    6.43E-05        0.00634249      4.62%   100.00% 3       65      ?a
?a  <worksAt>  ?f  ?f  <isLocatedIn>  ?b   => ?a  <livesIn>  ?b 4.287E-05       0.00422833      40.00%  100.00% 2       5       ?a
?a  <hasAcademicAdvisor>  ?f  ?f  <livesIn>  ?b   => ?a  <livesIn>  ?b  6.43E-05        0.00634249      25.00%  100.00% 3       12      ?a
?e  <hasAcademicAdvisor>  ?a  ?e  <livesIn>  ?b   => ?a  <livesIn>  ?b  6.43E-05        0.00634249      16.67%  100.00% 3       18      ?a
?a  <livesIn>  ?f  ?f  <isLocatedIn>  ?b   => ?a  <isPoliticianOf>  ?b  0.00023578      0.28947368      6.83%   100.00% 11      161     ?a
?a  <hasChild>  ?f  ?f  <isCitizenOf>  ?b   => ?a  <isCitizenOf>  ?b    4.287E-05       0.00464037      14.29%  100.00% 2       14      ?a
"""
# sample output:
"""
i_created(A,B) :- isKnownFor(A,B) {r0}.
i_isKnownFor(A,B) :- created(A,B) {r1}.
i_livesIn(A,B) :- isLeaderOf(A,B),wasBornIn(A,B) {r2}.
i_livesIn(A,B) :- wasBornIn(A,B),diedIn(A,B) {r3}.
i_livesIn(A,B) :- worksAt(A,F),isLocatedIn(F,B) {r4}.
i_livesIn(A,B) :- hasAcademicAdvisor(A,F),livesIn(F,B) {r5}.

rule    i_created
rule    r0
rule    i_isKnownFor
rule    r1
...
"""

import sys


def convertGoal(ret):
    # trim <> from <functor>
    # trim ? from ?a
    functor = ret[1][1:-1]
    return (functor, "%s(%s,%s)" % (functor,ret[0][1:].upper(),ret[2][1:].upper()))

def convertGoals(amie):
    ret = []
    for token in amie.split():
        ret.append(token)
        if len(ret)==3: 
            yield convertGoal(ret)
            ret = []
            
def convert(infn,outfnstem):
    with open(infn,'r') as f, open(outfnstem+".ppr",'w') as ppr, open(outfnstem+"-ruleids.cfacts","w") as cfacts:
        i=0
        for line in f:
            line = line.strip()
            # skip header
            if not line.startswith("?"): continue
            i+=1
            # also skip p(o,i) queries
            if not line.endswith("?a"): continue
            parts = line.split("\t")
            amieRule = parts[0]
            body,neck,head = amieRule.partition("=>")
            (functor,convertedHead) = convertGoal(head.split())
            ppr.write(" ".join(["i_"+convertedHead,":-",",".join([x[1] for x in convertGoals(body)]),"{r%d}.\n" % i]))
            cfacts.write("rule\ti_"+functor+"\n")
            cfacts.write("rule\tr%d\n" % i)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample')
    else:
        convert(sys.argv[1],sys.argv[2])
