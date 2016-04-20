# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import os
import os.path
import ops
import symtab 
import parser
import scipy.sparse
import scipy.io
import collections
import logging

# TODO index matEncoding by (pred,arity) pairs
# TODO use functor,arity pair names throughout

def assignGoal(var,const):
    return parser.Goal('assign',[var,const])

def isAssignMode(mode):
    """Is this a proper mode for the 'assign' predicate?"""
    if mode.arity==2 and mode.functor=='assign':
        assert mode.isOutput(0) and mode.isConst(1), 'proper usage for assign/2 is assign(Var,const) not %s' % mode
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
        self.stab.reservedSymbols.add("i")
        self.stab.reservedSymbols.add("o")
        # track if the matrix is arity one or arity two
        self.arity = {}
        #matEncoding[(functor,arity)] encodes predicate as a matrix
        self.matEncoding = {}
        #buffer data for a sparse matrix: buf[pred][i][j] = f
        #TODO: would lists and a coo matrix make a nicer buffer?
        def dictOfFloats(): return collections.defaultdict(float)
        def dictOfFloatDicts(): return collections.defaultdict(dictOfFloats)
        self.buf = collections.defaultdict(dictOfFloatDicts)
        #mark which matrices are 'parameters' by (functor,arity) pair
        self.params = {}

        
    #
    # retrieve matrixes, vectors, etc
    # 

    def dim(self):
        """Number of constants in the database, and dimension of all the vectors/matrices."""
        return self.stab.getMaxId() + 1

    def onehot(self,s):
        """A onehot row representation of a symbol."""
        assert self.stab.hasId(s),'constant %s not in db' % s
        n = self.dim()
        i = self.stab.getId(s)
        return scipy.sparse.csr_matrix( ([1.0],([0],[i])), shape=(1,n))

    def zeros(self):
        """An all-zeros row matrix."""
        n = self.dim()
        return scipy.sparse.csr_matrix( ([],([],[])), shape=(1,n))

    def ones(self):
        """An all-zeros row matrix."""
        n = self.dim()
        return scipy.sparse.csr_matrix( ([1]*n,([0]*n,[j for j in range(n)])), shape=(1,n))

    @staticmethod
    def transposeNeeded(mode,transpose=False):
        """Figure out if we should use the transpose of a matrix or not."""
        leftRight = (mode.isInput(0) and mode.isOutput(1))        
        return leftRight != transpose

    def matrix(self,mode,transpose=False):
        """The matrix associated with this mode - eg if mode is p(i,o) return
        a sparse matrix M_p so that v*M_p is appropriate for forward
        propagation steps from v.  
        """
        assert mode.arity==2,'arity of '+str(mode) + ' is wrong: ' + str(mode.arity)
        assert (mode.functor,mode.arity) in self.matEncoding,"can't find matrix for %s" % str(mode)
        if self.transposeNeeded(mode,transpose):
            return self.matEncoding[(mode.functor,mode.arity)]
        else:
            return self.matEncoding[(mode.functor,mode.arity)].transpose()            

    def matrixPreimage(self,mode):
        """The preimage associated with this mode, eg if mode is p(i,o) then
        return a row vector equivalent to 1 * M_p^T.  Also returns a row vector
        for a unary predicate."""
        if self.arity[mode.functor]==1:
            return self.matEncoding[(mode.functor,mode.arity)]
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
            return scipy.sparse.csr_matrix((data,(rowids,colids)),shape=(1,n))


    #
    # convert from vectors, matrixes to symbols - for i/o and debugging
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

    def matrixAsPredicateFacts(self,predicateFunctor,arity,m):
        result = {}
        m1 = scipy.sparse.coo_matrix(m)
        if arity==2:
            for i in range(len(m1.data)):
                a = self.stab.getSymbol(m1.row[i])
                b = self.stab.getSymbol(m1.col[i])
                w = m1.data[i]
                result[parser.Goal(predicateFunctor,[a,b])] = w
        else:
            assert arity==1
            for i in range(len(m1.data)):
                assert m1.row[i]==0
                b = self.stab.getSymbol(m1.col[i])
                w = m1.data[i]
                result[parser.Goal(predicateFunctor,[b])] = w
        return result
    #
    # i/o
    # TODO save/restore stab, matrices (separately)
    #

    def listing(self):
        for (functor,arity),m in self.matEncoding.items():
            print "DB: %s/%d" % (functor,arity)

    def serialize(self,dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        fp = open(os.path.join(dir,"symbols.txt"), 'w')
        for i in range(1,self.dim()):
            fp.write(self.stab.getSymbol(i) + '\n')
        fp.close()
        scipy.io.savemat(os.path.join(dir,"db.mat"),self.matEncoding,do_compression=True)
    
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
            arity = 2
            w = 1.0
        elif len(parts)==2:
            p,a1,a2 = parts[0],parts[1],None
            arity = 1
            w = 1.0
        else:
            logging.error("bad line '"+line+" '" + repr(parts)+"'")
            return
        self._checkArity(p,arity)
        if ((p,arity) in self.matEncoding):
            logging.error("predicate encoding is already completed for "+(p,arity)+ " at line: "+line)
            return
        i = self.stab.getId(a1)
        j = self.stab.getId(a2) if a2 else -1
        self.buf[(p,arity)][i][j] = w

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
                if not k%10000: logging.info('read %d lines' % k)
                self.bufferLine(line)

    def flushBuffers(self):
        """Flush all triples from the buffer."""
        for p,arity in self.buf.keys():
            self.flushBuffer(p,arity)

    def flushBuffer(self,p,arity):
        """Flush the triples defining predicate p from the buffer and define
        p's matrix encoding"""
        logging.info('flushing buffers for predicate %s' % p)
        n = self.stab.getMaxId() + 1
        if arity==2:
            m = scipy.sparse.lil_matrix((n,n))
            for i in self.buf[(p,arity)]:
                for j in self.buf[(p,arity)][i]:
                    m[i,j] = self.buf[(p,arity)][i][j]
            del self.buf[(p,arity)]
            self.matEncoding[(p,arity)] = scipy.sparse.csr_matrix(m)
            self.matEncoding[(p,arity)].sort_indices()
        elif arity==1:
            m = scipy.sparse.lil_matrix((1,n))
            for i in self.buf[(p,arity)]:
                for j in self.buf[(p,arity)][i]:
                    m[0,i] = self.buf[(p,arity)][i][j]
            del self.buf[(p,arity)]
            self.matEncoding[(p,arity)] = scipy.sparse.csr_matrix(m)
            self.matEncoding[(p,arity)].sort_indices()

    def clearBuffers(self):
        """Save space by removing buffers"""
        self.buf = None

    @staticmethod 
    def loadFile(filename):
        """Return a MatrixDB created by loading a files."""
        db = MatrixDB()
        db.bufferFile(filename)
        db.flushBuffers()
        db.clearBuffers()
        return db

    #
    # directly insert a matrix/vector
    #
    def insertPredicate(self,mat,functor,arity):
        self.arity[functor] = arity
        self.matEncoding[(functor,arity)] = mat
        self.matEncoding[(functor,arity)].sort_indices()
        (nrows,ncols) = mat.get_shape()
        assert (nrows==1 and arity==1) or (nrows==self.dim() and arity==2)
        return mat

    #
    # mark a predicate as a parameter
    # 
    def markAsParam(self,functor,arity):
        self.params[(functor,arity)] = self.matEncoding[(functor,arity)]

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

#
# test main
#

if __name__ == "__main__":
    if sys.argv[1]=='--serialize':
        print 'loading cfacts from ',sys.argv[2]
        db = MatrixDB.loadFile(sys.argv[2])
        print 'saving to',sys.argv[3]
        db.serialize(sys.argv[3])
    elif sys.argv[1]=='--deserialize':
        print 'loading saved db from ',sys.argv[2]
        db = MatrixDB.deserialize(sys.argv[2])
    elif sys.argv[1]=='--loadEcho':
        logging.basicConfig(level=logging.INFO)
        print 'loading cfacts from ',sys.argv[2]
        db = MatrixDB.loadFile(sys.argv[2])
        print db.matEncoding
        for (f,a),m in db.matEncoding.items():
            print f,a,m
            d = db.matrixAsPredicateFacts(f,a,m)
            print 'd for ',f,a,'is',d
            for k,w in d.items():
                print k,w
            
