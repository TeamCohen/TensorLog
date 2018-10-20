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

    htCount = Count(triples, by=lambda h_r_t7:(h_r_t7[0],h_r_t7[2]))
    rCount = Count(triples, by=lambda h_r_t8:h_r_t8[1])

    #
    #look for entailments: p(X,Y) :- q(X,Y)
    #
    candidateEntailments = Join( Jin(triples,by=lambda h_r_t:(h_r_t[0],h_r_t[2])), Jin(triples,by=lambda h_r_t1:(h_r_t1[0],h_r_t1[2]))) \
                           | Map( by=lambda h1_p_t1_h2_q_t2:((h1_p_t1_h2_q_t2[0][1],h1_p_t1_h2_q_t2[1][1]),(h1_p_t1_h2_q_t2[0][0],h1_p_t1_h2_q_t2[0][2]))) \
                           | Filter( by=lambda p_q_h_t:p_q_h_t[0][0]!=p_q_h_t[0][1]) \
                           | Count(by=lambda pq_ht:pq_ht[0])

    scoredEntailments = Join( Jin(candidateEntailments,by=lambda p_q_npq:p_q_npq[0][1]),Jin(rCount,lambda r_nr:r_nr[0])) \
                        | Map( by=lambda p__q_npq_q_nq : ((p__q_npq_q_nq[0][0],p__q_npq_q_nq[1][0]),math.log(p__q_npq_q_nq[0][1]/float(p__q_npq_q_nq[1][1]))) )


    entailmentsAsRules = Map(scoredEntailments, 
                             by=lambda p_q_score:'%s(X,Y):-%s(X,Y) {if_%s_%s}.\t#score %.3f' % (p_q_score[0][0],p_q_score[0][1],p_q_score[0][0],p_q_score[0][1],p_q_score[1]))

    #
    #look for inversions: p(X,Y) :- q(Y,X)
    #
    candidateInversions = Join( Jin(triples,by=lambda h_r_t2:(h_r_t2[0],h_r_t2[2])), Jin(triples,by=lambda h_r_t3:(h_r_t3[2],h_r_t3[0]))) \
                          | Map( by=lambda h1_p_t1_h2_q_t29:((h1_p_t1_h2_q_t29[0][1],h1_p_t1_h2_q_t29[1][1]),(h1_p_t1_h2_q_t29[0][0],h1_p_t1_h2_q_t29[0][2]))) \
                          | Count(by=lambda pq_ht10:pq_ht10[0])

    scoredInversions = Join( Jin(candidateInversions,by=lambda p_q_npq4:p_q_npq4[0][1]),Jin(rCount,lambda r_nr5:r_nr5[0])) \
                       | Map( by=lambda p__q_npq_q_nq11 : ((p__q_npq_q_nq11[0][0],p__q_npq_q_nq11[1][0]),math.log(p__q_npq_q_nq11[0][1]/float(p__q_npq_q_nq11[1][1]))) )

    inversionsAsRules = Map(scoredInversions,
                            by=lambda p_q_score12:'%s(X,Y):-%s(Y,X) {ifInv_%s_%s}.\t#score %.3f' % (p_q_score12[0][0],p_q_score12[0][1],p_q_score12[0][0],p_q_score12[0][1],p_q_score12[1]))

    #
    #look for chains: p(X,Y):-q(X,Z),r(Z,Y)
    #

    headToTail = Join( Jin(triples,by=lambda h1_r1_t1:h1_r1_t1[2]), Jin(triples,by=lambda h2_r2_t2:h2_r2_t2[0])) \
                 | Map(by=lambda h1_r1__mid_mid_r2_t2:(h1_r1__mid_mid_r2_t2[0][0],h1_r1__mid_mid_r2_t2[0][1],h1_r1__mid_mid_r2_t2[1][0],h1_r1__mid_mid_r2_t2[1][1],h1_r1__mid_mid_r2_t2[1][2]))
    
    candidateChains = Join( Jin(headToTail, by=lambda x_q_z_r_y:(x_q_z_r_y[0],x_q_z_r_y[4])), Jin(triples, by=lambda x_p_y:(x_p_y[0],x_p_y[2])) ) \
                      | Map(by=lambda x_q_z_r_y__x_p__y:((x_q_z_r_y__x_p__y[1][1],x_q_z_r_y__x_p__y[0][1],x_q_z_r_y__x_p__y[0][3]),(x_q_z_r_y__x_p__y[0][0],x_q_z_r_y__x_p__y[0][4]))) | Distinct() | Count(by=lambda pqr_xy:pqr_xy[0])

    qrCount = Map(headToTail, by=lambda x_q_z_r_y6:((x_q_z_r_y6[1],x_q_z_r_y6[3]),(x_q_z_r_y6[0],x_q_z_r_y6[4]))) | Count(by=lambda qr_xy:qr_xy[0])

    scoredChains = Join( Jin(candidateChains, by=lambda p_q_r_npqr:(p_q_r_npqr[0][1],p_q_r_npqr[0][2])), Jin(qrCount, by=lambda q_r_nqr:(q_r_nqr[0][0],q_r_nqr[0][1])) ) \
                   | Map(by=lambda p_q_r_npqr__q__r_nqr: ((p_q_r_npqr__q__r_nqr[0][0],p_q_r_npqr__q__r_nqr[0][1],p_q_r_npqr__q__r_nqr[0][2]),math.log(p_q_r_npqr__q__r_nqr[0][1]/float(p_q_r_npqr__q__r_nqr[1][1]))) )


    chainsAsRules = Map(scoredChains,
                        by=lambda p_q_r_score:'%s(X,Y):-%s(X,Z),%s(Z,Y) {chain_%s_%s_%s}.\t#score %.3f' % (p_q_r_score[0][0],p_q_r_score[0][1],p_q_r_score[0][2],p_q_r_score[0][0],p_q_r_score[0][1],p_q_r_score[0][2],p_q_r_score[1]))

    allRules = Union(entailmentsAsRules, inversionsAsRules, chainsAsRules) | Format()
    allRules.opts(storedAt='rules.ppr')

# always end like this
if __name__ == "__main__":
    planner = MineRules()
    planner.registerCompiler('mrs',gpextras.MRSCompiler)
    planner.main(sys.argv)
