# (C) William W. Cohen and Carnegie Mellon University, 2016

import scipy.sparse as SS
import numpy as np

#TODO rename to matutils
# miscellaneous broadcast utilities used my ops.py and funs.py

def mean(mat):
    """Return the average of the rows."""
    return SS.csr_matrix(mat.mean(axis=0))

def mapData(dataFun,mat):
    """Apply some function to the mat.data array of the sparse matrix and return a new one."""
    return SS.csr_matrix((dataFun(mat.data),mat.indices,mat.indptr), shape=mat.shape, dtype='float64')

def stack(mats):
    return SS.csr_matrix(SS.vstack(mats, dtype='float64'))

def numRows(m): 
    """Number of rows in matrix"""
    return m.get_shape()[0]

def numCols(m): 
    """Number of colunms in matrix"""
    return m.get_shape()[1]

def broadcastBinding(env,var1,var2):
    """Return a pair of shape-compatible matrices for the matrices stored
    in environment registers var1 and var2. """
    m1 = env[var1]
    m2 = env[var2]
    return broadcast2(m1,m2)

def broadcast2(m1,m2):
    """Return a pair of shape-compatible matrices for m1, m2 """
    r1 = numRows(m1)
    r2 = numRows(m2)
    if r1==r2:
        return m1,m2
    elif r1>1 and r2==1:
        return m1,stack([m2]*r1)
    elif r1==1 and r2>1:
        return stack([m1]*r2),m2

def broadcast4(m1a,m2a,m1b,m2b):
    """Given that the m1a,m2a and m1b,m2b are each pairwise compatible,
    broadcast to make all four matrices the same shape."""
    #print '=input sizes',[m.get_shape() for m in (m1a,m2a,m1b,m2b)]
    def broadcast(m,nr): return stack([m]*nr)
    r1 = numRows(m1a)
    r2 = numRows(m1b)
    if r1==r2:
        result = (m1a,m2a,m1b,m2b)
    elif r1==1:
        result = broadcast(m1a,r2),broadcast(m2a,r2),m1b,m2b
    elif r2==1:
        result = m1a,m2a,broadcast(m1b,r1),broadcast(m2b,r1)
    else:
        assert False,'broadcast fails'
    return result

#TODO unused?
def broadcastingDictSum(d1,d2,var):
    """ Given dicts d1 and d2, return the result of d1[var]+d2[var], after
    broadcasting, and defaulting missing values to zero.
    """
    if (var in d1) and (not var in d2):
        return d1[var]
    elif (var in d2) and (not var in d1):
        return d2[var]                
    else:
        def broadcast(m,r): return stack([m],r)
        r1 = d1[var].get_shape()[0]
        r2 = d2[var].get_shape()[0]
        if r1==r2:
            return d1[var] + d2[var]
        elif r1==1:
            return broadcast(d1[var], r2) + d2[var]
        elif r2==1:
            return d1[var] + broadcast(d2[var], r1)


#TODO unused?
def rowSum(m):
    """Sum of each row as a column vector."""
    numr = numRows(m)
    if numr==1:
        return m.sum()
    else:
        rows = [m.getrow(i) for i in range(numr)]
        return stack([r.sum() for r in rows])

def softmax(m):
    """Row-wise softmax of a sparse matrix, returned as a sparse matrix.
    This doesn't really require 'broadcasting' but it seems like you
    need special case handling to deal with multiple rows efficiently.
    """
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

    assert isinstance(m,SS.csr_matrix),'bad type for %r' % m
    numr = numRows(m)
    if numr==1:
        return softmaxRow(m)
    else:
        rows = [m.getrow(i) for i in range(numr)]
        return stack([softmaxRow(r) for r in rows])

def weightByRowSum(m1,m2):
    """Weight a rows of matrix m1 by the row sum of matrix m2."""
    r = numRows(m1)  #also m2
    if r==1:
        return  m1 * m2.sum()
    else:
        return SS.vstack([m2.getrow(i).sum() * m1.getrow(i) for i in range(r)], dtype='float64')
    
