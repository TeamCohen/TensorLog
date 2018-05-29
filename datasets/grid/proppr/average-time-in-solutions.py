import sys

# doesn't seem to work - maybe the times are bad
def scan_solutions():
    n = tot = 0
    for line in sys.stdin:
        if line[0]=='#':
            _rank,_goal,timeInMsec = line.strip().split("\t")
            (count,units) = timeInMsec.split(" ")
            assert units=="msec"
            n += 1
            tot += float(count)
    qps = 1000.0*n/tot
    print '==',sys.argv[1],'threads',sys.argv[2],'total',tot,'n',n,'average','%.2f' % (tot/n),'qps','%.2f' % qps

if __name__ == "__main__":
    for line in sys.stdin:
        if line.find("Total items:")>=0:
            _,n = line.strip().split(": ")
        elif line.find("Query-answering")>=0:
            _,t = line.strip().split(": ")
    qps = float(n)/(float(t)/1000)
    avg = float(t)/float(n)
    print '==',sys.argv[1],'threads',sys.argv[2],'total',t,'n',n,'average','%.2f' % avg,'qps','%.2f' % qps

