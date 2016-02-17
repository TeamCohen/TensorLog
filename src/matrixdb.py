# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import os
import os.path
import ops
import symtab 
import scipy.sparse
import scipy.io
import collections
import logging

# TODO: use lists in data_buf[p], row_buf[p], col_buf[p] to create a
# coo_matrix use coo_matrix.row, coo_matrix.col, coo_matrix.data, to
# serialize
#
# TODO: warn about constants i,o,i1,o1

def isSetMode(mode):
    """Is this a proper mode for the 'set' predicate?"""
    if mode.arity==2 and mode.functor=='set':
        assert mode.isOutput(0) and mode.isConst(1), 'proper usage for set/2 is set(Var,const) not %s' % mode
        return True
    else:
        return False

#
# a logical database implemented with sparse matrices
#

class MatrixDB(object):

    def __init__(self):
        #maps symbols to numeric ids
        self.stab = symtab.SymbolTable()
        # track if the matrix is arity one or arity two
        self.arity = {}
        #matEncoding[p] encodes predicate p as a matrix
        self.matEncoding = {}
        #preimageEncoding[mode] encodes the preimage of a binary predicate wrt a mode as a matrix
        self.preimageEncoding = {}
        #buffer data for a sparse matrix: buf[pred][i][j] = f
        def dictOfFloats(): return collections.defaultdict(float)
        def dictOfFloatDicts(): return collections.defaultdict(dictOfFloats)
        self.buf = collections.defaultdict(dictOfFloatDicts)
        
    #
    # retrieve matrixes, vectors, etc
    # 

    def dim(self):
        """Number of constants."""
        return self.stab.getMaxId() + 1

    def onehot(self,s):
        """A onehot row representation of a symbol."""
        assert self.stab.hasId(s),'constant %s not in db' % s
        n = self.dim()
        i = self.stab.getId(s)
        result = scipy.sparse.csr_matrix( ([1.0],([0],[i])), shape=(1,n))
        result[0,i] = 1.0
        return result

    def zeros(self):
        """An all-zeros row matrix."""
        n = self.dim()
        return scipy.sparse.csr_matrix( ([],([],[])), shape=(1,n))

    def matrix(self,mode,transpose=False):
        """The matrix associated with this mode - eg if mode is p(i,o) return
        a sparse matrix M_p so that v*M_p is appropriate for forward
        propagation steps from v.
        """
        assert mode.arity<=2,'arity of '+str(mode) + ' is too big: ' + str(mode.arity)
        assert mode.functor in self.matEncoding,"can't find matrix for %s" % str(mode)
        leftRight = (mode.isInput(0) and mode.isOutput(1))
        if leftRight != transpose:
            return self.matEncoding[mode.functor]
        else:
            return self.matEncoding[mode.functor].transpose()            

    def matrixPreimage(self,mode):
        """The preimage associated with this mode, eg if mode is p(i,o) then
        return a row vector equivalent to 1 * M_p^T.  Also returns a row vector
        for a unary predicate."""
        if mode in self.preimageEncoding:
            #caching
            return self.preimageEncoding[mode]
        else:
            if self.arity[mode.functor]==1:
                return self.matEncoding[mode.functor]
            else: 
                assert self.arity[mode.functor]==2
                #TODO mode is o,i vs i,o
                assert mode.isInput(0) and mode.isOutput(1)
                coo = self.matrix(mode).tocoo()
                rowsum = collections.defaultdict(float)
                for i in range(len(coo.data)):
                    r = coo.row[i]
                    d = coo.data[i]
                    rowsum[r] += d
                items = rowsum.items()
                data = [d for (r,d) in items]
                rowids = [0 for (r,d) in items]
                colids = [r for (r,d) in items]
                n = self.dim()
                pre = scipy.sparse.csc_matrix((data,(rowids,colids)),shape=(1,n))
                self.preimageEncoding[mode] = pre
                return self.preimageEncoding[mode]


    #
    # convert from vectors, matrixes to symbols
    # 

    def rowAsSymbolDict(self,row):
        result = {}
        coorow = row.tocoo()
        for i in range(len(coorow.data)):
            assert coorow.row[i]==0
            s = self.stab.getSymbol(coorow.col[i])
            result[s] = coorow.data[i]
        return result

    def matrixAsSymbolDict(self,m):
        result = {}
        (rows,cols)=m.shape
        for r in range(rows):
            result[r] = self.rowAsSymbolDict(m.getrow(r))
        return result

    #
    # i/o
    #TODO save/restore stab, matrices (separately)
    #

    def serialize(self,dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        fp = open(os.path.join(dir,"symbols.txt"), 'w')
        for i in range(1,self.dim()):
            fp.write(self.stab.getSymbol(i) + '\n')
        fp.close()
        scipy.io.savemat(os.path.join(dir,"db.mat"),db.matEncoding,do_compression=True)
    
    @staticmethod
    def deserialize(dir):
        db = MatrixDB()
        k = 1
        for line in open(os.path.join(dir,"symbols.txt")):
            i = db.stab.getId(line.strip())
            assert i==k,'symbols out of sync'
            k += 1
        scipy.io.loadmat(os.path.join(dir,"db.mat"),db.matEncoding)
        return db

    def _checkArity(self,p,k):
        if p in self.arity:
            assert self.arity[p]==k,'inconsistent number of arguments for ' + p + 'at line: '+repr(parts)
        else:
            self.arity[p]=k

    def bufferLine(self,line):
        """Load a single triple encoded as a tab-separated line.."""
        parts = line.split("\t")
        #TODO add weights
        if len(parts)==3:
            p,a1,a2 = parts[0],parts[1],parts[2]
            w = 1.0
            self._checkArity(p,2)
        elif len(parts)==2:
            p,a1 = parts[0],parts[1]
            w = 1.0
            a2 = None
            self._checkArity(p,1)
        else:
            logging.error("bad line '"+line+"'" + repr(parts))
            return
        if (p in self.matEncoding):
            logging.error("predicate encoding is already completed for "+p+ " at line: "+line)
            return
        i = self.stab.getId(a1)
        j = self.stab.getId(a2) if a2 else -1
        self.buf[p][i][j] = w

    def bufferLines(self,lines):
        """Load triples from a list of lines and buffer them internally"""
        for line in lines:
            loadLine(self,line.strip())

    def bufferFile(self,filename):
        """Load triples from a file and buffer them internally."""
        k = 0
        for line0 in open(filename):
            k += 1
            line = line0.strip()
            if line and (not line.startswith("#")):
                if not k%10000: print 'read',k,'lines'
                self.bufferLine(line)

    def flushBuffers(self):
        """Flush all triples from the buffer."""
        for p in self.buf.keys():
            self.flushBuffer(p)

    def flushBuffer(self,p):
        """Flush the triples defining predicate p from the buffer and define
        p's matrix encoding"""
        print 'flushing',p
        n = self.stab.getMaxId() + 1
        if self.arity[p]==2:
            m = scipy.sparse.lil_matrix((n,n))
            for i in self.buf[p]:
                for j in self.buf[p][i]:
                    m[i,j] = self.buf[p][i][j]
            del self.buf[p]
            self.matEncoding[p] = scipy.sparse.csc_matrix(m)
            self.matEncoding[p].sort_indices()
        elif self.arity[p]==1:
            m = scipy.sparse.lil_matrix((1,n))
            for i in self.buf[p]:
                for j in self.buf[p][i]:
                    m[0,i] = self.buf[p][i][j]
            del self.buf[p]
            self.matEncoding[p] = scipy.sparse.csr_matrix(m)
            self.matEncoding[p].sort_indices()

    def clearBuffers(self):
        self.buf = None

    @staticmethod 
    def loadFile(filename):
        """Return a MatrixDB created by loading a single file."""
        db = MatrixDB()
        db.bufferFile(filename)
        db.flushBuffers()
        db.clearBuffers()
        return db

    #
    # debugging
    # 
    #

    def dump(self):
        for p in self.matEncoding:
            print 'data   ',p,self.matEncoding[p].data
            print 'indices',p,self.matEncoding[p].indices
            print 'indptr ',p,self.matEncoding[p].indptr
        print "ids:"," ".join(self.stab.getSymbolList())

    def showFacts(self, rel):
        m = scipy.sparse.coo_matrix(fc.matEncoding[rel])
        for i in range(len(m.data)):
            print "\t".join([rel, self.stab.getSymbol(m.row[i]), self.stab.getSymbol(m.col[i]), str(m.data[i])])

#
# test main
#

if __name__ == "__main__":
    if not os.path.exists(sys.argv[2]):
        print 'loading cfacts from ',sys.argv[1]
        db = MatrixDB.loadFile(sys.argv[1])
        print 'saving to',sys.argv[2]
        db.serialize(sys.argv[2])
    else:
        print 'loading saved db from ',sys.argv[1]
        db = MatrixDB.deserialize(sys.argv[2])

    #test
    totDict = {}
    for s in sys.argv[3:]:
        tot = scipy.sparse.csr_matrix(([],([],[])),shape=(1,db.dim()))
        for t in s.split("+"):
            tot = tot + db.onehot(t)
        print tot
        print db.rowAsSymbolDict(tot)
        totDict[s] = tot
    for s1 in totDict.keys():
        for s2 in totDict.keys():
            if s1<s2:
                prod = totDict[s1].multiply(totDict[s2])
                print prod
                print s1,'*',s2,'=',db.rowAsSymbolDict(prod)
