import sys
import random

def generateGrid(n,outf):
    fp = open(outf,'w')
    for i in range(1,n+1):
        for j in range(1,n+1):
            for di in [-1,0,+1]:
                for dj in [-1,0,+1]:
                    if (0 <= i+di <= n) and (0 <= j+dj <= n) and (di!=0 or dj!=0):
                        fp.write('edge\tn_%d_%d\tn_%d_%d\t0.5\n' % (i,j,i+di,j+dj))

def generateData(n,trainFile,testFile):
    fpTrain = open(trainFile,'w')
    fpTest = open(testFile,'w')
    r = random.Random()
    for i in range(1,n+1):
        for j in range(1,n+1):
            #target
            ti = 1 if i<n/2 else n
            tj = 1 if j<n/2 else n
            x = 'n_%d_%d' % (i,j)
            y = 'n_%d_%d' % (ti,tj)
            fp = fpTrain if r.random()<0.67 else fpTest
            fp.write('\t'.join(['path',x,y]) + '\n')

if __name__=="__main__":
    n = int(sys.argv[1])
    outf = sys.argv[2]
    generateGrid(n,outf)
    if len(sys.argv)>3:
        generateData(n,sys.argv[3],sys.argv[4])

