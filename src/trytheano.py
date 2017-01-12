# (C) William W. Cohen and Carnegie Mellon University, 2016

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B
import matrixdb
import numpy

def debugVar(v,depth=0,maxdepth=10):
  if depth>maxdepth:
    print '...'
  else:
    print '| '*(depth+1),
    print 'var: name',v.name,'type',type(v),'def',theano.pp(v)
    for a in v.get_parents():
      debugApply(a,depth=depth+1,maxdepth=maxdepth)

def debugApply(a,depth=0,maxdepth=10):
  if depth>maxdepth:
    print '...'
  else:
    print '| '*(depth+1),
    print 'apply: ',a,'op',type(a.op),'output types',map(type,a.outputs)
    for v in a.inputs:
      debugVar(v,depth=depth+1,maxdepth=maxdepth)

if __name__=="__main__":

    db = matrixdb.MatrixDB.loadFile("test/fam.cfacts")
    va = db.onehot('william')
    vb = db.onehot('sarah')

    print 'a',va
    print 'b',vb
    print 'shape',va.shape

    print 'f1: s = x*((x+x)+x)'
    tx = S.csr_matrix('x')
    r1 = B.sp_sum(tx+tx+tx,sparse_grad=True)
    s = tx*r1
    s.name = 's'
    f1 = theano.function(inputs=[tx],outputs=[s])
    w = f1(va)
    print w[0]

    debugVar(s)

    #print db.rowAsSymbolDict(w[0])
#
#    print 'f2(w=a,c=b)'
#    tw = S.csr_matrix('w')  #weighter
#    tc = S.csr_matrix('c')  #constant
#    r2 = B.sp_sum(tw*1.7,sparse_grad=True)
#    s2 = tc*r2
#    f2 = theano.function(inputs=[tw,tc],outputs=[s2])
#    w = f2(va,vb)
#    print w[0]
#
    print 'f3(w=a), b constant'
    tw3 = S.csr_matrix('w')  #weighter
    #y = sparse.CSR(data, indices, indptr, shape)
#    tc3 = S.CSR(vb.data, vb.indices, vb.indptr, vb.shape)
#    r3 = B.sp_sum(tw3*1.7,sparse_grad=True)
#    s3 = tc3*r3
#    f3 = theano.function(inputs=[tw3],outputs=[s3])
#    w = f3(va)
#    print w[0]

#    debugVar(tw3,maxdepth=5)
