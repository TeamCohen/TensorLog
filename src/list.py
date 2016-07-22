import sys
import getopt

import declare
import tensorlog
import matrixdb

if __name__ == "__main__":
    def usage():
        print "usage: python -m list --db dbspec [--mode mode]"
    try:
        optlist,args = getopt.getopt(sys.argv[1:], 'x', ["db=","mode="])
    except getopt.GetoptError:
        usage()
        raise
    optdict = dict(optlist)
    if not '--db' in optdict:
        usage()
        assert False,'missing --db or --mode argument'
    db = tensorlog.parseDBSpec(optdict['--db'])
    if '--mode' in optdict:
        try:
            functor,rest = optdict['--mode'].split("/")
            arity = int(rest)
            m = db.matEncoding[(functor,arity)]
        except Exception:
            usage()
            assert False,'mode should be of the form functor/arity for something in the database'
        for goal,weight in db.matrixAsPredicateFacts(functor,arity,m).items():
            print '\t'.join([goal.functor] + goal.args + ['%g' % (weight)])
    else:
        db.listing()
