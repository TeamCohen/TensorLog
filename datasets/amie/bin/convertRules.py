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
    goal = "%s(%s,%s)" % (functor,ret[0][1:].upper(),ret[2][1:].upper())
    return (functor, goal, goal)


def convertGoals(amie):
    ret = []
    args = ['?a'] # always call rules with first argument bound
    build = []
    for token in amie.split():
        build.append(token)
        if len(build)==3: 
            gg = convertGoal(build)
            if build[0] in args and build[-1] in args:
                # TL does not handle this yet
                ret[0] = ("#%s" % ret[0][0],ret[0][1])

            if len(args)>0 and build[0] not in args and build[-1] in args:
                # switch to inverse
                #print "using inverse for %s\n\targs %s\n\tbody %s" % (gg[0]," ".join(args),amie)
                #print "\tret %s" % " ".join([r[1] for r in ret])
                
                ig = tuple(("inv_%s" % x for x in convertGoal(build[-1::-1])))
                gg = (ig[0],ig[1],gg[2])
                
                #gg = tuple(("inv_%s" % x for x in convertGoal(build[-1::-1])))
            
            # NB '?a' in args only works because no AMIE bodies have more than two goals
            if len(args)==0 or '?a' in args:
                ret.append(gg)
                args.append(build[0])
                args.append(build[-1])
            else: # bind variables left to right
                ret.insert(0,gg)
                args.append(build[0])
                args.append(build[-1])
            build = []
    return ret    

def convert(infn,outfnstem,prefix="",groundInverses=True):
    with open(infn,'r') as f, open(outfnstem+".ppr",'w') as ppr, open(outfnstem+"-ruleids.cfacts","w") as cfacts:
        ground = None
        if len(prefix)>0 or groundInverses:
            cfacts.write("rule\tground\n")
            ground = set()
        i=0
        for line in f:
            line = line.strip()
            # skip header
            if not line.startswith("?"): continue
            i+=1
            parts = line.split("\t")
            amieRule = parts[0]
            body,neck,head = amieRule.partition("=>")
            (functor,convertedHead,groundConvertedHead) = convertGoal(head.split())
            convertedBody = convertGoals(body)
            # TL does not handle p(A,B),r(A,B) sequences
            handled = not convertedBody[0][0].startswith("#")
            def rule(headfix="i_",tailfix=prefix):
                if not handled: ppr.write("# ")
                ppr.write(" ".join([headfix+convertedHead,":-",",".join([tailfix+x[1] for x in convertedBody]),"{r%d}.\n" % i]))
            rule()
            if handled:
                if len(prefix)>0:
                    for x,g,gg in convertedBody: 
                        if x not in ground:
                            ppr.write("i_{0} :- {1}{{ground}}.\n".format(g,gg if groundInverses else g))
                            ground.add(x)
                elif groundInverses:
                    for x,g,gg in convertedBody:
                        if x.startswith("inv_") and x not in ground:
                            ppr.write("{0} :- {1}{{ground}}.\n".format(g,gg))
                            ground.add(x)
            cfacts.write("rule\ti_"+functor+"\n") # just for the query, not the groundings -- used in a join later
            cfacts.write("rule\tr%d\n" % i)
            ppr.write("\n")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample')
    elif len(sys.argv) > 3:
        convert(*sys.argv[1:])
    else:
        convert(sys.argv[1],sys.argv[2])
