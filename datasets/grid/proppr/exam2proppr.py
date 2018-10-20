import sys

if __name__ == "__main__":
    for line in sys.stdin:
        rel,x,y = line.strip().split("\t")
        i,j = x.split(",")
        x = "%s_%s" % (i,j)
        print("%s(%s,Y)" % (rel,x))

