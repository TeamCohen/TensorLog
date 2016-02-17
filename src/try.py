# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import parser
import matrixdb
import tensorlog

#python try.py test/fam.cfacts 'p(i,o)' 'p(X,Y):-sister(X,Y) {r1}.' 'p(X,Y):-spouse(X,Y) {r2}.'

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print 'usage factfile mode rule1 ... x1 ... '
    else:
        db = matrixdb.MatrixDB.loadFile(sys.argv[1])
        mode = None
        rules = parser.RuleCollection()
        xs = []
        for a in sys.argv[2:]:
            if a.find(":-") >= 0:
                rules.add(parser.Parser.parseRule(a))
            elif a.find("(") >=0:
                assert mode==None, 'only one mode allowed'
                mode = tensorlog.ModeDeclaration(a)
            else:
                xs.append(a)
        w = 7*db.onehot('r1')+3*db.onehot('r2')
        p = tensorlog.ProPPRProgram(db=db,rules=rules,weights=w.transpose())
        p.ruleListing(mode)
        f = p.compile(mode)
        p.functionListing()

        for x in xs:
            print 'native result on input "%s":' % x
            result = p.eval(mode,[x])
            for val in result:
                print db.rowAsSymbolDict(val)
            print 'theano result on input "%s":' % x
            f = p.theanoPredictFunction(mode,['x'])
            result = f(db.onehot(x))
            for val in result:            
                #print val
                print db.rowAsSymbolDict(val)


