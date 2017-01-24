from guineapig import *
import gpextras
import sys
import math

#
# given a 'triples' file, which has tab-separated triples of the form
# (head,relation,tail) generate a file of all plausible rules and
# store in rules.ppr.  Uses GuineaPig package. Invocation for a small
# rule set:
#
#  python bin/minerules.py --params input:foo.triples --store allRules
#
# for a large one:
#
#  python -m mrs_gp --serve
#  python bin/minerules.py --opts viewdir:gpfs%3A,target:mrs,parallel:20,input:foo.triples --store allRules 
#  python -m mrs_gp --shutdown
# 

def Count(v=None, by=lambda x:x):
    return Group(inner=v, by=by, retaining=lambda row:1, reducingTo=ReduceToSum(), combiningTo=ReduceToSum())

class MineRules(Planner):

    D = GPig.getArgvParams(required=['input'])
    #read the triples (head,relation,tail)
    triples = ReadLines(D['input']) | Map(by=lambda line:line.strip().split("\t"))

    htCount = Count(triples, by=lambda(h,r,t):(h,t))
    rCount = Count(triples, by=lambda(h,r,t):r)

    #
    #look for entailments: p(X,Y) :- q(X,Y)
    #
    candidateEntailments = Join( Jin(triples,by=lambda(h,r,t):(h,t)), Jin(triples,by=lambda(h,r,t):(h,t))) \
                           | Map( by=lambda((h1,p,t1),(h2,q,t2)):((p,q),(h1,t1))) \
                           | Filter( by=lambda((p,q),(h,t)):p!=q) \
                           | Count(by=lambda(pq,ht):pq)

    scoredEntailments = Join( Jin(candidateEntailments,by=lambda((p,q),npq):q),Jin(rCount,lambda(r,nr):r)) \
                        | Map( by=lambda( ((p,_q),npq),(q,nq) ) : ((p,q),math.log(npq/float(nq))) )


    entailmentsAsRules = Map(scoredEntailments, 
                             by=lambda((p,q),score):'%s(X,Y):-%s(X,Y) {if_%s_%s}.\t#score %.3f' % (p,q,p,q,score))

    #
    #look for inversions: p(X,Y) :- q(Y,X)
    #
    candidateInversions = Join( Jin(triples,by=lambda(h,r,t):(h,t)), Jin(triples,by=lambda(h,r,t):(t,h))) \
                          | Map( by=lambda((h1,p,t1),(h2,q,t2)):((p,q),(h1,t1))) \
                          | Count(by=lambda(pq,ht):pq)

    scoredInversions = Join( Jin(candidateInversions,by=lambda((p,q),npq):q),Jin(rCount,lambda(r,nr):r)) \
                       | Map( by=lambda( ((p,_q),npq),(q,nq) ) : ((p,q),math.log(npq/float(nq))) )

    inversionsAsRules = Map(scoredInversions,
                            by=lambda((p,q),score):'%s(X,Y):-%s(Y,X) {ifInv_%s_%s}.\t#score %.3f' % (p,q,p,q,score))

    #
    #look for chains: p(X,Y):-q(X,Z),r(Z,Y)
    #

    headToTail = Join( Jin(triples,by=lambda(h1,r1,t1):t1), Jin(triples,by=lambda(h2,r2,t2):h2)) \
                 | Map(by=lambda((h1,r1,_mid),(mid,r2,t2)):(h1,r1,mid,r2,t2))
    
    candidateChains = Join( Jin(headToTail, by=lambda(x,q,z,r,y):(x,y)), Jin(triples, by=lambda(x,p,y):(x,y)) ) \
                      | Map(by=lambda((x,q,z,r,y),(_x,p,_y)):((p,q,r),(x,y))) | Distinct() | Count(by=lambda(pqr,xy):pqr)

    qrCount = Map(headToTail, by=lambda(x,q,z,r,y):((q,r),(x,y))) | Count(by=lambda(qr,xy):qr)

    scoredChains = Join( Jin(candidateChains, by=lambda((p,q,r),npqr):(q,r)), Jin(qrCount, by=lambda((q,r),nqr):(q,r)) ) \
                   | Map(by=lambda( ((p,q,r),npqr), ((_q,_r),nqr)): ((p,q,r),math.log(npqr/float(nqr))) )


    chainsAsRules = Map(scoredChains,
                        by=lambda((p,q,r),score):'%s(X,Y):-%s(X,Z),%s(Z,Y) {chain_%s_%s_%s}.\t#score %.3f' % (p,q,r,p,q,r,score))

    allRules = Union(entailmentsAsRules, inversionsAsRules, chainsAsRules) | Format()
    allRules.opts(storedAt='rules.ppr')

# always end like this
if __name__ == "__main__":
    planner = MineRules()
    planner.registerCompiler('mrs',gpextras.MRSCompiler)
    planner.main(sys.argv)
