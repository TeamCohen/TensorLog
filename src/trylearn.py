# working on a script now to compute the gradient and do learning.
# still some problems with type coercion in the interface between
# weights and theano....

# d = scipy.sparse.spdiags(x,0,n,n)
# returns a dia_matrix
# m.getnnz()
# d = scipy.sparse.spdiags(x,0,15,15,format='coo')
# matrices are often coerced to csr

#native mode seems to work for rows and matrices
#theano does not

#basic.py 
#  def sp_sum(x, axis=None, sparse_grad=False):
#  def mul(x, y):

NATIVE=False

import tensorlog

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B
import scipy.sparse

def loadExamples(file,db):
    xs = []
    ys = []
    for line in open(file):
        sx,sy = line.strip().split("\t")
        xs.append(db.onehot(sx))
        ys.append(db.onehot(sy))
    return xs,ys

p = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
p.setWeights(p.db.ones())
p.listing()

xs,ys = loadExamples("test/textcattoy-train.examples",p.db)
mode = tensorlog.ModeDeclaration('predict(i,o)')

