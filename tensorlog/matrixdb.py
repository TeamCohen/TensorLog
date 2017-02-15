# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# database abstraction which is based on sparse matrices
#

import sys
import os
import os.path
import scipy.sparse
import scipy.io
import collections
import logging
import numpy as NP

from tensorlog import config
from tensorlog import symtab
from tensorlog import parser
from tensorlog import mutil

conf = config.Config()
conf.allow_weighted_tuples = True; conf.help.allow_weighted_tuples = 'Allow last column of cfacts file to be a weight for the fact'

NULL_ENTITY_NAME = '__NULL__'

class MatrixParseError(Exception):
    def __init__(self,msg):
        self.msg = msg
    def __str__(self):
        return str(self.msg)

class MatrixFileError(Exception):
    def __init__(self,fname,line,p):
        self.filename=fname
        self.parseError=p
        self.line=line
    def __str__(self):
        return "on line %d of %s: %s" % (self.line,self.filename,str(self.parseError))

def assignGoal(var,const):
    """ A goal of the form assign(Y,const). """
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

    def __init__(self,stab=None):
        #maps symbols to numeric ids
        if not stab:
            self.stab = symtab.SymbolTable()
            self.stab.reservedSymbols.add("i")
            self.stab.reservedSymbols.add("o")
            self.stab.insert(NULL_ENTITY_NAME)
        else:
            self.stab = stab
        #matEncoding[(functor,arity)] encodes predicate as a matrix
        self.matEncoding = {}
        # buffer initialization: see startBuffers()
        #mark which matrices are 'parameters' by (functor,arity) pair
        self.params = set()

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
        return scipy.sparse.csr_matrix( ([float(1.0)],([0],[i])), shape=(1,n), dtype='float32')

    def zeros(self,numRows=1):
        """An all-zeros matrix."""
        n = self.dim()
        return scipy.sparse.csr_matrix( ([],([],[])), shape=(numRows,n), dtype='float32')

    def ones(self):
        """An all-ones row matrix."""
        n = self.dim()
        return scipy.sparse.csr_matrix( ([float(1.0)]*n,([0]*n,[j for j in range(n)])), shape=(1,n), dtype='float32')

    def nullMatrix(self,numRows=1):
        n = self.dim()
        nullId = self.stab.getId(NULL_ENTITY_NAME)
        return scipy.sparse.csr_matrix( ([float(1.0)]*numRows,
                                         (list(range(numRows)),[nullId]*numRows)),
                                        shape=(numRows,n),
                                        dtype='float32' )

    @staticmethod
    def transposeNeeded(mode,transpose=False):
        """Figure out if we should use the transpose of a matrix or not."""
        leftRight = (mode.isInput(0) and mode.isOutput(1))
        return leftRight == transpose

    def matrix(self,mode,transpose=False):
        """The matrix associated with this mode - eg if mode is p(i,o) return
        a sparse matrix M_p so that v*M_p is appropriate for forward
        propagation steps from v.
        """
        assert mode.arity==2,'arity of '+str(mode) + ' is wrong: ' + str(mode.arity)
        assert (mode.functor,mode.arity) in self.matEncoding,"can't find matrix for %s" % str(mode)
        if not self.transposeNeeded(mode,transpose):
            result = self.matEncoding[(mode.functor,mode.arity)]
        else:
            result = self.matEncoding[(mode.functor,mode.arity)].transpose()
            result = scipy.sparse.csr_matrix(result)
        mutil.checkCSR(result,'db.matrix mode %s transpose %s' % (str(mode),str(transpose)))
        return result

    def vector(self,mode):
        """Returns a row vector for a unary predicate."""
        assert mode.arity==1, "mode arity for '%s' must be 1" % mode
        result = self.matEncoding[(mode.functor,mode.arity)]
        return result

    def matrixPreimage(self,mode):
        """The preimage associated with this mode, eg if mode is p(i,o) then
        return a row vector equivalent to 1 * M_p^T.  Also returns a row vector
        for a unary predicate."""
        assert mode.arity==2, "mode arity for '%s' must be 2" % mode
        #TODO feels like this could be done more efficiently
        return self.ones() * self.matrix(mode,transpose=True)

    #
    # handling parameters
    #

    def isParameter(self,mode):
        return (mode.functor,mode.arity) in self.params

    def markAsParam(self,functor,arity):
        """ Mark a predicate as a parameter """
        self.params.add((functor,arity))

    def clearParamMarkings(self):
        """ Clear previously marked parameters"""
        self.params = set()

    def getParameter(self,functor,arity):
        assert (functor,arity) in self.params,'%s/%d not a parameter' % (functor,arity)
        return self.matEncoding[(functor,arity)]
        
    def parameterIsSet(self,functor,arity):
        return (functor,arity) in self.matEncoding

    def setParameter(self,functor,arity,replacement):
        assert (functor,arity) in self.params,'%s/%d not a parameter' % (functor,arity)
        self.matEncoding[(functor,arity)] = replacement

    #
    # convert from vectors, matrixes to symbols - for i/o and debugging
    # 

    def rowAsSymbolDict(self,row):
        result = {}
        coorow = row.tocoo()
        for i in range(len(coorow.data)):
            assert coorow.row[i]==0,"Expected 0 at coorow.row[%d]" % i
            s = self.stab.getSymbol(coorow.col[i])
            result[s] = coorow.data[i]
        return result

    def matrixAsSymbolDict(self,m):
        result = {}
        (rows,cols)=m.shape
        for r in range(rows):
            result[r] = self.rowAsSymbolDict(m.getrow(r))
        return result

    def matrixAsPredicateFacts(self,functor,arity,m):
        result = {}
        m1 = scipy.sparse.coo_matrix(m)
        if arity==2:
            for i in range(len(m1.data)):
                a = self.stab.getSymbol(m1.row[i])
                b = self.stab.getSymbol(m1.col[i])
                w = m1.data[i]
                result[parser.Goal(functor,[a,b])] = w
        else:
            assert arity==1,"Arity (%d) must be 1 or 2" % arity
            for i in range(len(m1.data)):
                assert m1.row[i]==0, "Expected 0 at m1.row[%d]" % i
                b = self.stab.getSymbol(m1.col[i])
                w = m1.data[i]
                result[parser.Goal(functor,[b])] = w
        return result


    #
    # query and display contents of database
    # 

    def inDB(self,functor,arity):
        return (functor,arity) in self.matEncoding

    def summary(self,functor,arity):
        m = self.matEncoding[(functor,arity)]
        return 'in DB: %s' % mutil.pprintSummary(m)

    def listing(self):
        for (functor,arity),m in sorted(self.matEncoding.items()):
            print '%s/%d: %s' % (functor,arity,self.summary(functor,arity))

    def numMatrices(self):
        return len(self.matEncoding.keys())

    def size(self):
        return sum(map(lambda m:m.nnz, self.matEncoding.values()))

    def parameterSize(self):
        return sum([m.nnz for  ((fun,arity),m) in self.matEncoding.items() if (fun,arity) in self.params])

    #
    # moving data between databases
    #

    def partnerWith(self,other):
        """Check that a database can be used as a partner.
        """
        assert other.dim()==self.dim(),"Dimensions don't match"

    def createPartner(self):
        """Create a 'partner' datavase, which shares the same symbol table,
        but not the same data. Matrices/relations can be moved back
        and forth between partners"""
        copy = MatrixDB(self.stab)
        return copy

    def copyToPartner(self,partner,functor,arity):
        partner.matEncoding[(functor,arity)] = self.matEncoding[(functor,arity)]
        if (functor,arity) in self.params:
            partner.params.add((functor,arity))

    def moveToPartner(self,partner,functor,arity):
        self.copyToPartner(partner,functor,arity)
        if (functor,arity) in self.params:
            self.params.remove((functor,arity))
        del self.matEncoding[(functor,arity)]

    
    #TODO not clear if this is the right place for this logic
    def matrixAsTrainingData(self,functor,arity):
        """ Convert a matrix containing pairs x,f(x) to training data for a
        learner.  For each row x with non-zero entries, copy that row
        to Y, and and also append a one-hot representation of x to the
        corresponding row of X.
        """
        xrows = []
        yrows = []
        m = self.matEncoding[(functor,arity)].tocoo()
        n = self.dim()
        for i in range(len(m.data)):
            x = m.row[i]
            xrows.append(scipy.sparse.csr_matrix( ([float(1.0)],([0],[x])), shape=(1,n) ))
            rx = m.getrow(x)
            yrows.append(rx * (float(1.0)/rx.sum()))
        return mutil.stack(xrows),mutil.stack(yrows)

    #
    # i/o
    #

    def serialize(self,direc):
        if not os.path.exists(direc):
            os.makedirs(direc)
        fp = open(os.path.join(direc,"symbols.txt"), 'w')
        for i in range(1,self.dim()):
            fp.write(self.stab.getSymbol(i) + '\n')
        fp.close()
        scipy.io.savemat(os.path.join(direc,"db.mat"),self.matEncoding,do_compression=True)

    @staticmethod
    def deserialize(direc):
        db = MatrixDB()
        k = 1
        for line in open(os.path.join(direc,"symbols.txt")):
            i = db.stab.getId(line.strip())
            assert i==k,'symbols out of sync for symbol "%s": expected index %d actual %d' % (line.strip(),i,k)
            k += 1
        scipy.io.loadmat(os.path.join(direc,"db.mat"),db.matEncoding)
        #serialization/deserialization ends up converting
        #(functor,arity) pairs to strings and csr_matrix to csc_matrix
        #so convert them back....
        for stringKey,mat in db.matEncoding.items():
            del db.matEncoding[stringKey]
            if not stringKey.startswith('__'):
                db.matEncoding[eval(stringKey)] = scipy.sparse.csr_matrix(mat)
        logging.info('deserialized database has %d relations and %d non-zeros' % (db.numMatrices(),db.size()))
        return db

    @staticmethod
    def uncache(dbFile,factFile):
        """Build a database file from a factFile, serialize it, and return
        the de-serialized database.  Or if that's not necessary, just
        deserialize it.  As always the factFile can be a
        colon-separated list.
        """
        if not os.path.exists(dbFile) or any([os.path.getmtime(f)>os.path.getmtime(dbFile) for f in factFile.split(":")]):
            db = MatrixDB.loadFile(factFile)
            db.serialize(dbFile)
            os.utime(dbFile,None) #update the modification time for the directory
            return db
        else:
            logging.info('deserializing db file '+ dbFile)
            return MatrixDB.deserialize(dbFile)


    def bufferLine(self,line):
        """Load a single triple encoded as a tab-separated line.."""
        def atof(s):
            try:
                return float(s)
            except ValueError:
                return float(0.0)

        parts = line.split("\t")
        if conf.allow_weighted_tuples and len(parts)==4:
            f,a1,a2,wstr = parts[0],parts[1],parts[2],parts[3]
            arity = 2
            w = atof(wstr)
        elif len(parts)==3:
            f,a1,a2 = parts[0],parts[1],parts[2]
            w = atof(a2)
            if not conf.allow_weighted_tuples or w==0:
                arity = 2
                w = float(1.0)
            else:
                arity = 1
                #w is ok still
        elif len(parts)==2:
            f,a1,a2 = parts[0],parts[1],None
            arity = 1
            w = float(1.0)
        else:
            logging.error("bad line '"+line+" '" + repr(parts)+"'")
            return
        key = (f,arity)
        if (key in self.matEncoding):
            logging.error("predicate encoding is already completed for "+str(key)+ " at line: "+line)
            return
        i = self.stab.getId(a1)
        j = self.stab.getId(a2) if a2 else -1
        try:
            self.buf[key][i][j] = w
        except TypeError as e:
            raise MatrixParseError(e)

    def bufferLines(self,lines):
        """Load triples from a list of lines and buffer them internally"""
        for line in lines:
            self.bufferLine(line) #was: loadLine (undefined?)

    def bufferFile(self,filename):
        """Load triples from a file and buffer them internally."""
        k = 0
        for line0 in open(filename):
            k += 1
            line = line0.strip()
            if line and (not line.startswith("#")):
                if not k%10000: logging.info('read %d lines' % k)
                try:
                    self.bufferLine(line)
                except MatrixParseError as e:
                    raise MatrixFileError(filename,k,e)

    def flushBuffers(self):
        """Flush all triples from the buffer."""
        for f,arity in self.buf.keys():
            self.flushBuffer(f,arity)

    def flushBuffer(self,f,arity):
        """Flush the triples defining predicate p from the buffer and define
        p's matrix encoding"""
        logging.info('flushing %d buffered facts for predicate %s' % (len(self.buf[(f,arity)]),f))

        n = self.stab.getMaxId() + 1
        if arity==2:
            m = scipy.sparse.lil_matrix((n,n),dtype='float32')
            for i in self.buf[(f,arity)]:
                for j in self.buf[(f,arity)][i]:
                    m[i,j] = self.buf[(f,arity)][i][j]
            del self.buf[(f,arity)]
            self.matEncoding[(f,arity)] = scipy.sparse.csr_matrix(m,dtype='float32')
            self.matEncoding[(f,arity)].sort_indices()
        elif arity==1:
            m = scipy.sparse.lil_matrix((1,n))
            for i in self.buf[(f,arity)]:
                for j in self.buf[(f,arity)][i]:
                    m[0,i] = self.buf[(f,arity)][i][j]
            del self.buf[(f,arity)]
            self.matEncoding[(f,arity)] = scipy.sparse.csr_matrix(m,dtype='float32')
            self.matEncoding[(f,arity)].sort_indices()
        mutil.checkCSR(self.matEncoding[(f,arity)], 'flushBuffer %s/%d' % (f,arity))

    def rebufferMatrices(self):
        """Re-encode previously frozen matrices after a symbol table update"""
        n = self.stab.getMaxId() + 1
        for (functor,arity),m in self.matEncoding.items():
            (rows,cols) = m.get_shape()
            if cols != n:
                logging.info("Re-encoding predicate %s" % functor)
                if arity==2:
                    # first shim the extra rows
                    shim = scipy.sparse.lil_matrix((n-rows,cols))
                    m = scipy.sparse.vstack([m,shim])
                    (rows,cols) = m.get_shape()
                # shim extra columns
                shim = scipy.sparse.lil_matrix((rows,n-cols))
                self.matEncoding[(functor,arity)] = scipy.sparse.hstack([m,shim],format="csr")
                self.matEncoding[(functor,arity)].sort_indices()

    def clearBuffers(self):
        """Save space by removing buffers"""
        self.buf = None

    def startBuffers(self):
        #buffer data for a sparse matrix: buf[pred][i][j] = f
        #TODO: would lists and a coo matrix make a nicer buffer?
        def dictOfFloats(): return collections.defaultdict(float)
        def dictOfFloatDicts(): return collections.defaultdict(dictOfFloats)
        self.buf = collections.defaultdict(dictOfFloatDicts)

    def addLines(self,lines):
        self.startBuffers()
        self.bufferLines(lines)
        self.rebufferMatrices()
        self.flushBuffers()
        self.clearBuffers()

    def addFile(self,filename):
        logging.info('adding cfacts file '+ filename)
        self.startBuffers()
        self.bufferFile(filename)
        self.rebufferMatrices()
        self.flushBuffers()
        self.clearBuffers()

    @staticmethod 
    def loadFile(filenames):
        """Return a MatrixDB created by loading a file.  Also allows a
        colon-separated list of files
        """
        db = MatrixDB()
        for f in filenames.split(":"):
            db.addFile(f)
        logging.info('loaded database has %d relations and %d non-zeros' % (db.numMatrices(),db.size()))
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

#
# test main
#

if __name__ == "__main__":
    if sys.argv[1]=='--serialize':
        print 'loading cfacts from',sys.argv[2]
        if sys.argv[2].find(":")>=0:
            db = MatrixDB()
            for f in sys.argv[2].split(":"):
                db.addFile(f)
        else:
            db = MatrixDB.loadFile(sys.argv[2])
        print 'saving to',sys.argv[3]
        db.serialize(sys.argv[3])
    elif sys.argv[1]=='--deserialize':
        print 'loading saved db from ',sys.argv[2]
        db = MatrixDB.deserialize(sys.argv[2])
    elif sys.argv[1]=='--uncache':
        print 'uncaching facts',sys.argv[3],'from',sys.argv[2]
        db = MatrixDB.uncache(sys.argv[2],sys.argv[3])
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
