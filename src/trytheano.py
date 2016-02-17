# (C) William W. Cohen and Carnegie Mellon University, 2016

import theano
import theano.tensor as T
import theano.sparse as S
import theano.sparse.basic as B
import matrixdb
import numpy

if __name__=="__main__":
    db = matrixdb.MatrixDB.loadFile("test/fam.cfacts")
    va = db.onehot('william')
    vb = db.onehot('sarah')

    print 'a',va
    print 'b',vb
    print 'shape',va.shape

    print 'f1'
    tx = S.csr_matrix('x')
    r1 = B.sp_sum(tx+tx+tx,sparse_grad=True)
    s = tx*r1
    f1 = theano.function(inputs=[tx],outputs=[s])
    w = f1(va)
    print w[0]
    #print db.rowAsSymbolDict(w[0])
    
    print 'f2(w=a,c=b)'
    tw = S.csr_matrix('w')  #weighter
    tc = S.csr_matrix('c')  #constant
    r2 = B.sp_sum(tw*1.7,sparse_grad=True)    
    s2 = tc*r2
    f2 = theano.function(inputs=[tw,tc],outputs=[s2])
    w = f2(va,vb)
    print w[0]

    print 'f3(w=a), b constant'
    tw3 = S.csr_matrix('w')  #weighter
    #y = sparse.CSR(data, indices, indptr, shape)
    tc3 = S.CSR(vb.data, vb.indices, vb.indptr, vb.shape)
    r3 = B.sp_sum(tw3*1.7,sparse_grad=True)    
    s3 = tc3*r3
    f3 = theano.function(inputs=[tw3],outputs=[s3])
    w = f3(va)
    print w[0]


#    r2 = B.sp_sum(tw+tw,sparse_grad=True)
#    #tb = S.csc_matrix(vb,name='tb')  #constant
#    bcoo = vb.tocoo()
#    tb = S.csc_matrix((bcoo.data,bcoo.row,bcoo.col),name='tb')
#    s2 = tb*r2
#    #s2 = r1*va



#
#    print 'f2'
#    r2 = tx+vb
#    f2 = theano.function(inputs=[tx],outputs=[r2])
#    w = f2(va)
#    print w[0]
#
#    print 'f3'
#    r2 = B.row_scale(tx, r1)
#    f2 = theano.function(inputs=[tx],outputs=[r2])
#    w = f2(va)
#    print w[0]

