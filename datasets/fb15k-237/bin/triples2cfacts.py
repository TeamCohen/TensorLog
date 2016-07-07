import sys
import collections
import random

if __name__=="__main__":
    # usage: foo.clean > foo.cfacts
    tripleFile = sys.argv[1]
    for line in open(tripleFile):
        (head,rel,tail) = line.strip().split("\t")
        print '\t'.join([rel,head,tail])

