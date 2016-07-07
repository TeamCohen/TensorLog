import sys
import collections
import random

if __name__=="__main__":
    # usage: train.cfacts foo.triples > foo.exam
    factFile = sys.argv[1]
    tripleFile = sys.argv[2]
    dbConst = set()

    for line in open(factFile):
        (rel,head,tail) = line.strip().split("\t")
        dbConst.add(head) 
        dbConst.add(tail) 

    yValues = collections.defaultdict(list)
    for line in open(tripleFile):
        (head,rel,tail) = line.strip().split("\t")
        yValues[(rel,head)].append(tail)

    dropped = 0
    for (r,x),ys in yValues.items():
        if x not in dbConst:
            dropped += 1
        else:
            ys1 = filter(lambda y:y in dbConst, ys)
            if not ys1:
                dropped += 1
            else:
                print '\t'.join(['i_'+r,x]+ys1)




