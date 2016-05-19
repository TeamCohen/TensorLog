import sys

def generateGrid(n,outf):
    fp = open(outf,'w')
    for i in range(1,n+1):
        for j in range(1,n+1):
            for di in [-1,0,+1]:
                for dj in [-1,0,+1]:
                    if (0 <= i+di <= n) and (0 <= j+dj <= n) and (di!=0 or dj!=0):
                        fp.write('edge\tn_%d_%d\tn_%d_%d\t0.5\n' % (i,j,i+di,j+dj))

if __name__=="__main__":
    n = int(sys.argv[1])
    outf = sys.argv[2]
    generateGrid(n,outf)
