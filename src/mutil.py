# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import scipy.io
import numpy as np
import math

import config

conf = config.Config()
conf.optimize_softmax = True;   conf.help.optimize_softmax = 'use optimized version of softmax code'

# miscellaneous broadcast utilities used my ops.py and funs.py

def summary(mat):
    _checkCSR(mat)
    return 'nnz %d rows %d cols %d' % (mat.nnz,numRows(mat),numCols(mat))

def _checkCSR(mat):
    assert isinstance(mat,SS.csr_matrix),'bad type for %r' % mat

def mean(mat):
    """Return the average of the rows."""
    _checkCSR(mat)
    #TODO - mat.mean returns a dense matrix which mutil converts, can I avoid that?
    return SS.csr_matrix(mat.mean(axis=0))

def mapData(dataFun,mat):
    """Apply some function to the mat.data array of the sparse matrix and return a new one."""
    _checkCSR(mat)
    def showMat(msg,m): print msg,type(m),m.shape
    newdata = dataFun(mat.data)
    assert newdata.shape==mat.data.shape,'shape mismatch %r vs %r' % (newdata.shape,mat.data.shape)
    return SS.csr_matrix((newdata,mat.indices,mat.indptr), shape=mat.shape, dtype='float64')

def nzCols(mat,rowIndex):
    """Enumerate the non-zero column indices in row i."""
    _checkCSR(mat)
    for colIndex in mat.indices[mat.indptr[rowIndex]:mat.indptr[rowIndex+1]]:
        yield colIndex

def emptyRow(mat,rowIndex):
    _checkCSR(mat)
    return mat.indptr[rowIndex]==mat.indptr[rowIndex+1]

def stack(mats):
    """Vertically stack matrices and return a sparse csr matrix."""
    for m in mats: _checkCSR(m)
    return SS.csr_matrix(SS.vstack(mats, dtype='float64'))

def numRows(m): 
    """Number of rows in matrix"""
    _checkCSR(m)
    return m.get_shape()[0]

def numCols(m): 
    """Number of colunms in matrix"""
    _checkCSR(m)
    return m.get_shape()[1]

def broadcastBinding(env,var1,var2):
    """Return a pair of shape-compatible matrices for the matrices stored
    in environment registers var1 and var2. """
    m1 = env[var1]
    m2 = env[var2]
    _checkCSR(m1); _checkCSR(m2)
    return broadcast2(m1,m2)

def broadcast2(m1,m2):
    """Return a pair of shape-compatible matrices for m1, m2 """
    _checkCSR(m1); _checkCSR(m2)
    r1 = numRows(m1)
    r2 = numRows(m2)
    if r1==r2:
        return m1,m2
    elif r1>1 and r2==1:
        return m1,stack([m2]*r1)
    elif r1==1 and r2>1:
        return stack([m1]*r2),m2
    else:
        assert False,'cannot broadcast: #rows %d vs %d' % (r1,r2)

def softmax(db,m):
    """Row-wise softmax of a sparse matrix, returned as a sparse matrix.
    This doesn't really require 'broadcasting' but it seems like you
    need special case handling to deal with multiple rows efficiently.
    """
    _checkCSR(m)
    nullEpsilon = -10  # scores for null entity will be exp(nullMatrix)
    def softmaxRow(r):
        if not r.nnz:
            # evals to uniform
            n = numCols(r)
            return SS.csr_matrix( ([1.0/n]*n,([0]*n,[j for j in range(n)])), shape=(1,n))
        else:
            d = r.data
            e_d = np.exp(d - np.max(d))
            #TODO should I correct the denominator for the (r.numCols()-r.nnz) zeros in the row?
            #that would be: d_sm = e_d / (e_d.sum() + numCols(r) - r.nnz)
            d_sm = e_d / e_d.sum()
            return SS.csr_matrix((d_sm,r.indices,r.indptr),shape=r.shape)
    numr = numRows(m)
    if numr==1:
        return softmaxRow(m + db.nullMatrix(1)*nullEpsilon)
    elif not conf.optimize_softmax:
        m1 = m + db.nullMatrix(numr)*nullEpsilon
        rows = [m1.getrow(i) for i in range(numr)]
        return stack([softmaxRow(r) for r in rows])
    else:
        result = m + db.nullMatrix(numr)*nullEpsilon
        for i in xrange(numr):
            #rowMax = max(result[i,j] for j in nzCols(result,i))
            rowMax = max(result.data[result.indptr[i]:result.indptr[i+1]])
            rowNorm = 0
            #for j in nzCols(result,i):
            for j in range(result.indptr[i],result.indptr[i+1]):
                #result[i,j] = math.exp(result[i,j] - rowMax)
                #rowNorm += result[i,j]
                result.data[j] = math.exp(result.data[j] - rowMax)
                rowNorm += result.data[j]
            #for j in nzCols(result,i):            
            for j in range(result.indptr[i],result.indptr[i+1]):
                #result[i,j] = result[i,j]/rowNorm
                result.data[j] = result.data[j] / rowNorm
        return result

def broadcastAndComponentwiseMultiply(m1,m2):
    _checkCSR(m1); _checkCSR(m2)
    def multiplyByBroadcastRowVec(r,m,v):
        vd = {}
        for j in range(v.indptr[0],v.indptr[1]):
            vd[v.indices[j]] = v.data[j]
        result = m1.copy()
        for i in xrange(r):
            #for j in nzCols(m,i):
            #    result[i,j] = result[i,j]*v[0,j]
            for j in range(result.indptr[i],result.indptr[i+1]):
                k = result.indices[j]
                if k in vd:
                    result.data[j] = result.data[j] * vd[k]
                else:
                    result.data[j] = 0.0
        return result
    r1 = numRows(m1)
    r2 = numRows(m2)
    if r1==r2:
        return  m1.multiply(m2)
    elif r1==1:
        return multiplyByBroadcastRowVec(r2,m2,m1)
    elif r2==1:
        return multiplyByBroadcastRowVec(r1,m1,m2)        
    else:
        assert False, 'mismatched matrix sizes: #rows %d,%d' % (r1,r2)

def broadcastAndWeightByRowSum(m1,m2):
    _checkCSR(m1); _checkCSR(m2)
    """ Optimized combination of broadcast2 and weightByRowSum operations
    """ 
    r1 = numRows(m1)
    r2 = numRows(m2)
    if r2==1:
        return  m1 * m2.sum()
    elif r1==1 and r2>1:
        n = numCols(m1)
        nnz1 = m1.data.shape[0]
        #allocate space for the broadcast version of m1,
        #with one copy of m1 for every row of m2
        data = np.zeros(shape=(nnz1*r2,))
        indices = np.zeros(shape=(nnz1*r2,),dtype='int')
        indptr = np.zeros(shape=(r2+1,),dtype='int')
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


def weightByRowSum(m1,m2):
    """Weight a rows of matrix m1 by the row sum of matrix m2."""
    _checkCSR(m1); _checkCSR(m2)
    r = numRows(m1)  #also m2
    assert numRows(m2)==r
    if r==1:
        return  m1 * m2.sum()
    else:
        #old slow version 
        #  return SS.vstack([m2.getrow(i).sum() * m1.getrow(i) for i in range(r)], dtype='float64')
        #optimized
        result = m1.copy()
        for i in xrange(r):
            w = m2.data[m2.indptr[i]:m2.indptr[i+1]].sum()
            result.data[result.indptr[i]:result.indptr[i+1]] *= w
        return result

if __name__=="__main__":
    testmat = {}
    scipy.io.loadmat("test.mat",testmat)
    m1 = testmat['m1']
    m2 = testmat['m2']
    broadcastAndWeightByRowSum(m1,m2)

