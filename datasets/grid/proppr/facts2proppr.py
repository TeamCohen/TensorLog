import sys

if __name__ == "__main__":
    for line in sys.stdin:
        rel,x,y,w = line.strip().split("\t")
        i,j = x.split(",")
        x = "%s_%s" % (i,j)
        i,j = y.split(",")
        y = "%s_%s" % (i,j)
        print "\t".join([rel,x,y])


