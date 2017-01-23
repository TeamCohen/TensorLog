# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# miscellaneous matrix utilities
#

import scipy.sparse as SS
import scipy.io
import numpy as NP
import numpy.random as NR
import math
import logging

import config
import matrixdb

conf = config.Config()
conf.careful = True;                 conf.help.careful = 'execute checks for matrix type and NANs'
conf.densifyWeightByRowSum = False;  conf.help.densifyWeightByRowSum = 'use dense matrices here - did not speed up test cases'
conf.densifyMean = True;             conf.help.densifyMean = 'use dense matrices here'
conf.maxExpandFactor = 3;            conf.help.maxExpand = 'K, where you can can use B + KM the sparse-matrix memory M when densifying matrices'
conf.maxExpandIntercept = 10000;     conf.help.maxExpand = 'B, where you can can use B + KM the sparse-matrix memory M when densifying matrices'
conf.warnAboutDensity = False;       conf.help.warnAboutDensity = 'warn when you fail to densify a matrix'

NP.seterr(all='raise',under='ignore')
# stop execution & print traceback for various floating-point issues
# except underflow; aiui we don't mind if very small numbers go to zero --kmm

# comparison to None in scipy is switching to elementwise so we're going to check type instead
NONETYPE=type(None) 

def summary(mat):
    """Helpful string describing a matrix for debugging."""
    checkCSR(mat)
    return 'nnz %d rows %d cols %d' % (mat.nnz,numRows(mat),numCols(mat))

def pprintSummary(mat):
    if type(mat)!=NONETYPE:
        checkCSR(mat)
        return '%3d x %3d [%d nz]' % (numRows(mat),numCols(mat),mat.nnz)
    else:
        return '___'

def checkCSR(mat,context='unknown'):
    """Raise error if mat is not a scipy.sparse.csr_matrix."""
    if conf.careful:
        assert isinstance(mat,SS.csr_matrix),'bad type [context %s] for %r' % (context,mat)

def checkNoNANs(mat,context='unknown'):
    """Raise error if mat has nan's in it"""
    if conf.careful:
        checkCSR(mat)
        assert not NP.any(NP.isnan(mat.data)), 'nan\'s found: %s' % context

def maxValue(mat):
    try:
        return NP.max(mat.data)
    except ValueError:
        #zero-size array
        return -1

def densify(mat,maxExpandFactor=-1,maxExpandIntercept=-1):
    """Create a smallish dense version of a sparse matrix, which slices
    out the range of columns which have non-zero values, and return a pair
    D,I where D is the dense matrix, and I is information needed to
    invert the process for a matrix with the same dimensions.  Returns
    None if the dense matrix would be too much larger.
    """
    if maxExpandFactor<0: maxExpandFactor = conf.maxExpandFactor
    if maxExpandIntercept<0: maxExpandIntercept = conf.maxExpandIntercept

    hiIndex = NP.max(mat.indices)
    loIndex = NP.min(mat.indices)
    ds = denseSize(mat,loIndex,hiIndex)
    ss = sparseSize(mat,loIndex,hiIndex)
    if  ds > ss*maxExpandFactor + maxExpandIntercept:
        if conf.warnAboutDensity: logging.warn('no expansion: sparse size only %d dense size is %d' % (ss,ds))
        return None,None
    else:
        newShape = (numRows(mat),hiIndex-loIndex+1)
        D = SS.csr_matrix((mat.data,mat.indices-loIndex,mat.indptr),shape=newShape,dtype='float64').todense()
        return D,(loIndex,numCols(mat))

def denseSize(m,loIndex,hiIndex):
    return (hiIndex-loIndex) * numRows(m)

def sparseSize(m,loIndex,hiIndex):
    return numRows(m)+1 + 2*m.nnz

def codensify(m1,m2,maxExpandFactor=-1,maxExpandIntercept=-1):
    """ Similar to densify but returns a triple with two dense matrices and an 'info' object.
    """
    assert numCols(m1)==numCols(m2),"Cannot codensify matrices with different number of columns"
    if m1.nnz==0 or m2.nnz==0:
        return None,None,None
    if maxExpandFactor<0: maxExpandFactor = conf.maxExpandFactor
    if maxExpandIntercept<0: maxExpandIntercept = conf.maxExpandIntercept
    loIndex = min(NP.min(m1.indices),NP.min(m2.indices))
    hiIndex = max(NP.max(m1.indices),NP.max(m2.indices))
    ds = denseSize(m1,loIndex,hiIndex)+denseSize(m2,loIndex,hiIndex)
    ss = sparseSize(m1,loIndex,hiIndex)+sparseSize(m2,loIndex,hiIndex)
    if ds > (ss * maxExpandFactor + maxExpandIntercept):
        if conf.warnAboutDensity: logging.warn('no expansion: sparse size only %d dense size is %d' % (ss,ds))
        return None,None,None
    newShape1 = (numRows(m1),hiIndex-loIndex+1)
    newShape2 = (numRows(m2),hiIndex-loIndex+1)
    D1 = SS.csr_matrix((m1.data,m1.indices-loIndex,m1.indptr),shape=newShape1,dtype='float64').todense()
    D2 = SS.csr_matrix((m2.data,m2.indices-loIndex,m2.indptr),shape=newShape2,dtype='float64').todense()
    return D1,D2,(loIndex,numCols(m1))

def undensify(denseMat, info):
    loIndex,numCols = info
    (numRows,_) = denseMat.shape
    tmp = SS.csr_matrix(denseMat)
    result = SS.csr_matrix((tmp.data,tmp.indices+loIndex,tmp.indptr),shape=(numRows,numCols),dtype='float64')
    result.eliminate_zeros()
    return result

def mean(mat):
    """Return the average of the rows in a matrix."""
    checkCSR(mat)
    return SS.csr_matrix(mat.mean(axis=0))
#    r = numRows(mat)
#    return rowsum(mat) * (1.0/r)

def rowsum(mat):
    """Return the sum of the rows in a matrix."""
    checkCSR(mat)
    # mat.sum(0) returns a dense matrix, and using these incantations will avoid that
    # v1: squish everything into one row and sum duplicates - slower than csr_matrix(mat.mean)
#    newIndptr = NP.array([0,mat.data.shape[0]])
#    rowSum = SS.csr_matrix((mat.data, mat.indices, newIndptr),(1,numCols(mat)),dtype='float64')
#    rowSum.sum_duplicates() # modifies in-place
    # v2: use rowsum[k] = sum_{j:indices[j]==k} data[j] and turn it into a matrix dot product
    # still 2x slower than dense mat.mean
#    ndense = mat.data.shape[0]
#    indptr2 = NP.arange(0,ndense+1)
#    m2 = SS.csr_matrix((mat.data,mat.indices,indptr2),(ndense,numCols(mat)))
#    sparseOnes = SS.csr_matrix((NP.ones(ndense),NP.arange(0,ndense),NP.array([0,ndense])), (1,ndense), dtype='float64')
#    rowSum = sparseOnes.dot(m2)
    # v3: like v2, but densify
    denseMat,undensifier = densify(mat)
    if type(denseMat)!=NONETYPE:
        return undensify(denseMat.sum(0), undensifier)
    else:
        ndense = mat.data.shape[0]
        indptr2 = NP.arange(0,ndense+1)
        m2 = SS.csr_matrix((mat.data,mat.indices,indptr2),(ndense,numCols(mat)))
        sparseOnes = SS.csr_matrix((NP.ones(ndense),NP.arange(0,ndense),NP.array([0,ndense])), (1,ndense), dtype='float64')
        rowSum = sparseOnes.dot(m2)
        return rowSum

def mapData(dataFun,mat):
    """Apply some function to the mat.data array of the sparse matrix and return a new one."""
    checkCSR(mat)
    newdata = dataFun(mat.data)
    return SS.csr_matrix((newdata,mat.indices,mat.indptr), shape=mat.shape, dtype='float64')

#TODO avoid this, it's expensive
def stack(mats):
    """Vertically stack matrices and return a sparse csr matrix."""
    for m in mats: checkCSR(m)
    return SS.csr_matrix(SS.vstack(mats, dtype='float64'))

def numRows(m): 
    """Number of rows in matrix"""
    checkCSR(m)
    return m.shape[0]

def numCols(m): 
    """Number of colunms in matrix"""
    checkCSR(m)
    return m.shape[1]

def nzCols(m,i):
    """Enumerate the non-zero columns in row i."""
    for j in range(m.indptr[i],m.indptr[i+1]):
        yield j

def repeat(row,n):
    """Construct an n-row matrix where each row is a copy of the given one."""
    checkCSR(row)
    assert numRows(row)==1,"Tried to repeat multi-row matrix"
    #create the data and indices vector - which are just n copies of
    #the row data
    d = NP.tile(row.data,n)
    inds = NP.tile(row.indices,n)
    #create the indptr
    numNZCols = row.indptr[1]
    if numNZCols>0:
        ptrs = NP.array(range(0,numNZCols*n+1,numNZCols))
    else:
        ptrs = NP.zeros(n+1, dtype='int')
    return SS.csr_matrix((d,inds,ptrs),shape=(n,numCols(row)), dtype='float64')


def alterMatrixRows(mat,alterationFun):
    """ apply alterationFun(data,lo,hi) to each row.
    """
    for i in range(numRows(mat)):
        alterationFun(mat.data,mat.indptr[i],mat.indptr[i+1],mat.indices)

def softmax(db,mat):
    """ Compute the softmax of each row of a matrix.
    """
    nullEpsilon = -10  # scores for null entity will be exp(nullEpsilon)
    result = repeat(db.nullMatrix(1)*nullEpsilon, numRows(mat)) + mat
    denseResult,undensifier = densify(result)
    if type(denseResult)!=NONETYPE:
        return undensify(denseSoftmax(denseResult), undensifier)
    else:
        def softMaxAlteration(data,lo,hi,unused):
            rowMax = max(data[lo:hi])
            assert not math.isnan(rowMax),"softMaxAlteration: NaN rowMax"
            data[lo:hi] = NP.exp(data[lo:hi] - rowMax)
            rowNorm = sum(data[lo:hi])
            assert not math.isnan(rowNorm),"softMaxAlteration: NaN rowNorm"
            data[lo:hi] /= rowNorm
            #replace the zeros in data, which are underflow, with something small
            minValue = math.exp(nullEpsilon)
            segment = data[lo:hi]
            segment[segment==0] = minValue
            data[lo:hi] = segment
        alterMatrixRows(result,softMaxAlteration)
        return result

def denseSoftmax(m):
    #we want to make sure we keep the zero entries as zero
    mask = m!=0
    e_m = NP.multiply(NP.exp(m - m.max(axis=1)), mask)
    return e_m / e_m.sum(axis=1)

def broadcastAndComponentwiseMultiply(m1,m2):
    """ compute m1.multiply(m2), but broadcast m1 or m2 if necessary
    """
    checkCSR(m1); checkCSR(m2)
    r1 = numRows(m1); r2 = numRows(m2)
    if r1==r2:
        return  m1.multiply(m2)
    else:
        assert r1==1 or r2==1, 'mismatched matrix sizes: #rows %d,%d' % (r1,r2)
    if r1==1:
        return multiplyByBroadcastRowVec(m1,m2)
    else:
        return multiplyByBroadcastRowVec(m1,m2)

def multiplyByBroadcastRowVec(m,v):
    (dm,dv,i) = codensify(m,v)
    if type(dm)!=NONETYPE:
        dp = NP.multiply(dm,dv)
        return undensify(dp, i)
    else:
        bv = repeat(v, numRows(m))
        return m.multiply(bv)

#TODO: this is slow - about 2/3 of learning time
def broadcastAndWeightByRowSum(m1,m2):
    checkCSR(m1); checkCSR(m2)
    """ Optimized combination of broadcast2 and weightByRowSum operations
    """
    if conf.densifyWeightByRowSum:
        (d1,d2,i) = codensify(m1, m2)
        if type(d1)!=NONETYPE:
            dr = NP.multiply(d1, d2.sum(axis=1))
            return undensify(dr, i)

    r1 = numRows(m1)
    r2 = numRows(m2)
    if r2==1:
        return  m1 * m2.sum()
    elif r1==1 and r2>1:
        bm1 = repeat(m1, r2)
        for i in xrange(r2):
            w = m2.data[m2.indptr[i]:m2.indptr[i+1]].sum()
            bm1.data[bm1.indptr[i]:bm1.indptr[i+1]] = m1.data * w
        return bm1
    else:
        assert r1==r2, "broadcastAndWeightByRowSum: r1 must match r2"
        result = m1.copy()
        for i in xrange(r1):
            w = m2.data[m2.indptr[i]:m2.indptr[i+1]].sum()
            result.data[result.indptr[i]:result.indptr[i+1]] *= w
        return result

def shuffleRows(m,shuffledRowNums=None):
    """Create a copy of m with the rows permuted."""
    checkCSR(m)
    if type(shuffledRowNums)==NONETYPE:
        shuffledRowNums = NP.arange(numRows(m))
        NR.shuffle(shuffledRowNums)
    data = NP.array(m.data)
    indices = NP.array(m.indices)
    indptr = NP.array(m.indptr)
    lo = 0
    for i in range(m.indptr.size-1 ):
        r = shuffledRowNums[i]
        rowLen = m.indptr[r+1] - m.indptr[r]
        indptr[i] = lo
        indptr[i+1] = lo + rowLen
        lo += rowLen
        for j in range(rowLen):
            data[indptr[i]+j] = m.data[m.indptr[r]+j]
            indices[indptr[i]+j] = m.indices[m.indptr[r]+j]
    result = SS.csr_matrix((data,indices,indptr), shape=m.shape, dtype='float64')
    result.sort_indices()
    return result

def selectRows(m,lo,hi):
    """Return a sparse matrix that copies rows lo...hi-1 of m.  If hi is
    too large it will be adjusted. """
    checkCSR(m)
    if hi>numRows(m): hi=numRows(m)
    #data for rows [lo, hi) are in cells [jLo...jHi)
    jLo = m.indptr[lo]
    jHi = m.indptr[hi]
    #allocate space
    data = NP.zeros(jHi - jLo)
    indices = NP.zeros(jHi - jLo, dtype='int')
    indptr = NP.zeros(hi - lo + 1, dtype='int')
    for i in range(hi - lo):
        rowLen = m.indptr[lo+i+1] - m.indptr[lo+i]
        #translate the index pointers
        indptr[i] = m.indptr[lo+i] - jLo
        for j in range(rowLen):
            k = m.indptr[lo+i]+j
            data[indptr[i] + j] = m.data[k]
            indices[indptr[i] + j] = m.indices[k]
    indptr[hi-lo] = m.indptr[hi] - jLo
    result = SS.csr_matrix((data,indices,indptr), shape=(hi-lo,numCols(m)), dtype='float64')
    return result

if __name__=="__main__":
    tmp = []
    for i in range(1,11):
        tmp.append([i] + [0]*3 + [5*i])
    m = SS.csr_matrix(tmp)
    print m.todense()
    m2 = shuffleRows(m)
    #print m2.todense()
    for i in range(0,10,4):
        print selectRows(m2,i,i+4).todense()
