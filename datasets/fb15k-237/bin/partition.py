import sys
import random

if __name__=="__main__":
    # usage: inputs percent output1 output2
    inputfile = sys.argv[1]
    percent = float(sys.argv[2])
    outfile1 = sys.argv[3]
    outfile2 = sys.argv[4]
    rnd = random.Random()
    fp1 = open(outfile1,'w')
    fp2 = open(outfile2,'w')
    for line in open(inputfile):
        if 100*rnd.random() < percent:
            fp1.write(line)
        else:
            fp2.write(line)            
