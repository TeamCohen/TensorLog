# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# miscellaneous matrix utilities
#

import scipy.sparse as SS
import scipy.io
import numpy as NP
import math
import logging

import config

conf = config.Config()
conf.optimize_softmax = True;   conf.help.optimize_softmax = 'use optimized version of softmax code'

# miscellaneous broadcast utilities used my ops.py and funs.py

NP.seterr(all='raise',under='ignore') 
# stop execution & print traceback for various floating-point issues
# except underflow; aiui we don't mind if very small numbers go to zero --kmm

def summary(mat):
    """Helpful string describing a matrix for debugging.""" 
    checkCSR(mat)
    return 'nnz %d rows %d cols %d' % (mat.nnz,numRows(mat),numCols(mat))

def checkCSR(mat,context='unknown'):
    """Raise error if mat is not a scipy.sparse.csr_matrix."""
    assert isinstance(mat,SS.csr_matrix),'bad type [context %s] for %r' % (context,mat)

def checkNoNANs(mat,context='unknown'):
    """Raise error if mat has nan's in it"""
    checkCSR(mat)
    for j in range(0,mat.indptr[-1]):
        assert not math.isnan(mat.data[j]),'nan\'s found: %s' % context

def maxValue(mat):
    try:
        return NP.max(mat.data)
    except ValueError:
        #zero-size array
        return -1

def mean(mat):
    """Return the average of the rows."""
    checkCSR(mat)
    #TODO - mat.mean returns a dense matrix which mutil converts, can I avoid that?
    #TODO - does this need broadcasting?
    return SS.csr_matrix(mat.mean(axis=0))

def mapData(dataFun,mat,selector=None,default=0):
    """Apply some function to the mat.data array of the sparse matrix and return a new one."""
    checkCSR(mat)
    def showMat(msg,m): print msg,type(m),m.shape
    dat = mat.data

    # FIXME: indptr isn't the same shape as indices! indptr maps
    # row->indices range, so if we mean to remove some of the things
    # in indices, indptr is going to get allllll messed up

    selected = None
    if selector: 
        selected = selector(mat.data)
        dat = dat[selected]
    newdata = dataFun(dat)
    if selector:
        buf = np.ones_like(mat.data) * default
        buf[selected] = newdata
        newdata = buf

    assert newdata.shape==mat.data.shape,'shape mismatch %r vs %r' % (newdata.shape,mat.data.shape)
    return SS.csr_matrix((newdata,mat.indices,mat.indptr), shape=mat.shape, dtype='float64')

def stack(mats):
    """Vertically stack matrices and return a sparse csr matrix."""
    for m in mats: checkCSR(m)
    return SS.csr_matrix(SS.vstack(mats, dtype='float64'))

def nzCols(m,i):
    """Enumerate the non-zero columns in row i."""
    for j in range(m.indptr[i],m.indptr[i+1]):
        yield j

def repeat(row,n):
    """Construct an n-row matrix where each row is a copy of the given one."""
    checkCSR(row)
    d = NP.tile(row.data,n)
    inds = NP.tile(row.indices,n)
    assert numRows(row)==1
    numNZCols = row.indptr[1]
    ptrs = NP.array(range(0,numNZCols*n+1,numNZCols))
    return SS.csr_matrix((d,inds,ptrs),shape=(n,numCols(row)), dtype='float64')

def alterMatrixRows(mat,alterationFun):
    """ apply alterationFun(data,lo,hi) to each row.
    """
    for i in range(numRows(mat)):
        alterationFun(mat.data,mat.indptr[i],mat.indptr[i+1],mat.indices)

def softmax(db,mat):
    """ Compute the softmax of each row of a matrix.
    """
    nullEpsilon = -10  # scores for null entity will be exp(nullMatrix)
    result = repeat(db.nullMatrix(1)*nullEpsilon, numRows(mat)) + mat
    def softMaxAlteration(data,lo,hi,unused):
        rowMax = max(data[lo:hi])
        assert not math.isnan(rowMax)
        for j in range(lo,hi):
            data[j] = math.exp(data[j] - rowMax)
        rowNorm = sum(data[lo:hi])
        assert not math.isnan(rowNorm)
        for j in range(lo,hi):
            data[j] = data[j]/rowNorm
            assert not math.isnan(data[j])
            if data[j]==0:
                data[j] = math.exp(nullEpsilon)
    alterMatrixRows(result,softMaxAlteration)
    return result

def broadcastAndComponentwiseMultiply(m1,m2):
    """ compute m1.multiply(m2), but broadcast m1 or m2 if necessary
    """
    checkCSR(m1); checkCSR(m2)
    def multiplyByBroadcastRowVec(m,v):
        #convert v to a dictionary
        vd = dict( (v.indices[j],v.data[j]) for j in range(v.indptr[0],v.indptr[1]) )
        def multiplyByVAlteration(data,lo,hi,indices):
            for j in range(lo,hi):
                data[j] *= vd.get(indices[j],0.0)
        result = m.copy()
        alterMatrixRows(result,multiplyByVAlteration)
        return result
    r1 = numRows(m1); r2 = numRows(m2)
    if r1==r2:
        return  m1.multiply(m2)
    elif r1==1:
        return multiplyByBroadcastRowVec(m1,m2)        
    elif r2==1:
        return multiplyByBroadcastRowVec(m1,m2)        
    else:
        assert False, 'mismatched matrix sizes: #rows %d,%d' % (r1,r2)
    return result

def numRows(m): 
    """Number of rows in matrix"""
    checkCSR(m)
    return m.get_shape()[0]

def numCols(m): 
    """Number of colunms in matrix"""
    checkCSR(m)
    return m.get_shape()[1]

def broadcastAndWeightByRowSum(m1,m2):
    checkCSR(m1); checkCSR(m2)
    """ Optimized combination of broadcast2 and weightByRowSum operations
    """ 
    r1 = numRows(m1)
    r2 = numRows(m2)
    if r2==1:
        try:
            return  m1 * m2.sum()
        except FloatingPointError:
            print "broadcastAndWeightByRowSum m1: %s" % summary(m1)
            print "broadcastAndWeightByRowSum m2.sum(): %s" % m2.sum()
            raise
    elif r1==1 and r2>1:
        n = numCols(m1)
        nnz1 = m1.data.shape[0]
        #allocate space for the broadcast version of m1,
        #with one copy of m1 for every row of m2
        data = NP.zeros(shape=(nnz1*r2,))
        indices = NP.zeros(shape=(nnz1*r2,),dtype='int')
        indptr = NP.zeros(shape=(r2+1,),dtype='int')
        ptr = 0
        indptr[0] = 0 
        for i in xrange(r2):
            w = m2.data[m2.indptr[i]:m2.indptr[i+1]].sum()
            #multiply the non-zero datapoints by w and copy them into the right places
            for j in xrange(nnz1):
                data[ptr+j] = m1.data[j]*w
                indices[ptr+j] = m1.indices[j]
            # increment the indptr so indptr[i]:indptr[i+1] tells
            # where to find the data, indices for row i
            indptr[i+1]= indptr[i]+nnz1
            ptr += nnz1
        result = SS.csr_matrix((data,indices,indptr),shape=(m2.shape[0],m2.shape[1]), dtype='float64')
        return result
    else:
        assert r1==r2
        result = m1.copy()
        for i in xrange(r1):
            w = m2.data[m2.indptr[i]:m2.indptr[i+1]].sum()
            result.data[result.indptr[i]:result.indptr[i+1]] *= w
        return result

if __name__=="__main__":
    db = matrixdb.MatrixDB.uncache('tlog-cache/textcat.db','test/textcattoy.cfacts')
    m = prog.db.matEncoding[('posPair',2)]
    
