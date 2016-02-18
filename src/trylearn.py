# working on a script now to compute the gradient and do learning.
# still some problems with type coercion in the interface between
# weights and theano....

import tensorlog

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B

def loadExamples(file,db):
    xs = []
    ys = []
    for line in open(file):
        sx,sy = line.strip().split("\t")
        xs.append(db.onehot(sx))
        ys.append(db.onehot(sy))
    return B.vstack(xs),B.vstack(ys)


p = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
p.setWeights(p.db.ones())

p.listing()

X,Y = loadExamples("test/textcattoy-train.examples",p.db)
mode = tensorlog.ModeDeclaration('predict(i,o)')
#f = p.theanoPredictFunction(mode,['x'])
#prediction = f(X)
prediction = p.eval(mode,[X])

print db.matrixAsSymbolDict(prediction)





