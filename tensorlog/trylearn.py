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

#
# set up the program
#

p = tensorlog.ProPPRProgram.load(["test/textcat.ppr","test/textcattoy.cfacts"])
p.setWeights(p.db.ones())
p.listing()

#
# load the data
#

xs,ys = loadExamples("test/textcattoy-train.examples",p.db)

#returns inputs and outputs that are used to build the prediction
#function
mode = tensorlog.ModeDeclaration('predict(i,o)')
ins,outs = p.theanoPredictExpr(mode,['x'])
scorex = outs[0]  #the actual score vector for x

# something simple to try differentiating
toyLoss = B.sp_sum(scorex,sparse_grad=True)
print 'gradToyLoss...'
gradToyLoss = T.grad(toyLoss, p.getParams())


#
# now define a theano function that computes loss for ONE example
#
y = S.csr_matrix('y')
prob = scorex * (1.0/B.sp_sum(scorex, sparse_grad=True))        #scale x to 0-1
loss = B.sp_sum(-y * B.structured_log(prob),sparse_grad=True)   #cross-entropy loss
print 'loss...'
theano.printing.debugprint(loss)
lossFun = theano.function(inputs=[ins[0],y], outputs=[loss])

#
# test one one example
#
lossFunResult = lossFun(xs[0],ys[0])
print 'loss on example 0:',lossFunResult[0]

#
# compute gradient
#

#this is where things fail now
#  File "/Library/Python/2.7/site-packages/theano/gradient.py", line 1262, in access_grad_cache
#    str(node.op), term.ndim, var.ndim))
#ValueError: MulSD.grad returned a term with 2 dimensions, but 0 are required.
#

print 'gradLoss...'
gradLoss = T.grad(loss, p.getParams())



