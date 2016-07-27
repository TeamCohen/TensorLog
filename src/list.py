import sys
import getopt

#
# 
# 

import declare
import tensorlog
import matrixdb

if __name__ == "__main__":
    def usage():
        print "usage: python -m list --db dbspec [--mode mode]"
        print "  without mode specified: lists the relations in the database"
        print "  with mode specified: lists the facts in one relation in .cfacts format"
        print "usage: python -m list --prog progspec [--ruleIds]"
        print "  list the all rule ids"
    argspec = ["db=","mode=","prog=","ruleIds"]
    try:
        optlist,args = getopt.getopt(sys.argv[1:], 'x', argspec)
    except getopt.GetoptError:
        usage()
        raise
    optdict = dict(optlist)

    db = tensorlog.parseDBSpec(optdict['--db']) if '--db' in optdict else None
    if db and (not '--mode' in optdict):
        db.listing()
    elif db and ('--mode' in optdict):
        try:
            functor,rest = optdict['--mode'].split("/")
            arity = int(rest)
            m = db.matEncoding[(functor,arity)]
            for goal,weight in db.matrixAsPredicateFacts(functor,arity,m).items():
                print '\t'.join([goal.functor] + goal.args + ['%g' % (weight)])
        except Exception:
            usage()
            assert False,'mode should be of the form functor/arity for something in the database'
    elif '--prog' in optdict:
        prog = tensorlog.parseProgSpec(optdict['--prog'],db,proppr=True)
        for rid in prog.ruleIds:
            print '\t'.join(['ruleid',rid])
    else:
        usage()

