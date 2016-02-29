# (C) William W. Cohen and Carnegie Mellon University, 2016

class Envir(object):
    def __init__(self,db):
        self.binding = {}
        self.db = db


#python try.py test/fam.cfacts 'p(i,o)' 'p(X,Y):-sister(X,Y) {r1}.' 'p(X,Y):-spouse(X,Y) {r2}.'

if __name__ == "__main__":
    e = Envir('db')
    e['foo'] = 'bar'
    print e['foo']

