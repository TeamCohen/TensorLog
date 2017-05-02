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

from tensorlog import config
from tensorlog import declare
from tensorlog import schema
from tensorlog import parser
from tensorlog import mutil

conf = config.Config()
conf.allow_weighted_tuples = True;     conf.help.allow_weighted_tuples = 'Allow last column of cfacts file to be a weight for the fact'
conf.default_to_typed_schema = False;  conf.help.default_to_typed_schema = 'If true use TypedSchema() as default schema in MatrixDB'
conf.ignore_types = False;             conf.help.ignore_types = 'Ignore type declarations, even if they are present'

NULL_ENTITY_NAME = schema.NULL_ENTITY_NAME
THING = schema.THING
#functor in declarations of trainable relations, eg trainable(posWeight,1)
TRAINABLE_DECLARATION_FUNCTOR = 'trainable'

class MatrixDB(object):
  """ A logical database implemented with sparse matrices """

  def __init__(self,initSchema=None):
    #matEncoding[(functor,arity)] encodes predicate as a matrix
    self.matEncoding = {}
    # mark which matrices are 'parameters' by (functor,arity) pair
    self.paramSet = set()
    self.paramList = []
    # buffers for reading in facts in tab-sep form
    self._databuf = self._rowbuf = self._colbuf = None
    if initSchema is not None:
      self.schema = initSchema
    elif conf.default_to_typed_schema and not conf.ignore_types:
      self.schema = schema.TypedSchema()
    else:
      self.schema = schema.UntypedSchema()

  def checkTyping(self):
    self.schema.checkTyping(self.matEncoding.keys())

  def isTypeless(self):
    return self.schema.isTypeless()

  #
  # retrieve matrixes, vectors, etc
  #

  def _fillDefault(self,typeName):
    return self.schema.defaultType() if typeName is None else typeName

  def dim(self,typeName=None):
    typeName = self._fillDefault(typeName)
    """Number of constants in the database, and dimension of all the vectors/matrices."""
    return self.schema.getMaxId(typeName) + 1

  def onehot(self,s,typeName=None,outOfVocabularySymbolsAllowed=False):
    typeName = self._fillDefault(typeName)
    """A onehot row representation of a symbol."""
    if outOfVocabularySymbolsAllowed and not self.schema.hasId(typeName,s):
        return self.onehot(schema.OOV_ENTITY_NAME,typeName)
    assert self.schema.hasId(typeName,s),'constant %s (type %s) not in db' % (s,typeName)
    n = self.dim(typeName)
    i = self.schema.getId(typeName,s)
    return scipy.sparse.csr_matrix( ([float(1.0)],([0],[i])), shape=(1,n), dtype='float32')

  def zeros(self,numRows=1,typeName=None):
    typeName = self._fillDefault(typeName)
    """An all-zeros matrix."""
    n = self.dim(typeName)
    return scipy.sparse.csr_matrix( ([],([],[])), shape=(numRows,n), dtype='float32')

  def ones(self,typeName=None):
    """An all-ones row matrix."""
    typeName = self._fillDefault(typeName)
    n = self.dim(typeName)
    return scipy.sparse.csr_matrix( ([float(1.0)]*n,([0]*n,[j for j in range(n)])), shape=(1,n), dtype='float32')

  def nullMatrix(self,numRows=1,typeName=None,numCols=0):
    """A matrix where every row is a one-hot encoding of the null entity.
    The number of columns is specified by numCols or by
    a typeName.  If numCols==0 and typeName==None then
    use numCols=dim(THING)
    """
    if typeName is None: typeName = THING
    if numCols==0: numCols = self.dim(typeName)
    nullId = 1
    return scipy.sparse.csr_matrix( ([float(1.0)]*numRows,
                                     (list(range(numRows)),[nullId]*numRows)),
                                    shape=(numRows,numCols),
                                    dtype='float32' )

  @staticmethod
  def transposeNeeded(mode,transpose=False):
    """For mode x, which is p(i,o) or p(o,i), considers the matrix M=M_x
    if transpose==False and M=M_x.transpose() if transpose is True.
    Returns False if M is self.matEncoding[(p,2)] and True if M is
    self.matEncoding[(p,2)].transpose()
    """
    leftRight = (mode.isInput(0) and mode.isOutput(1))
    return leftRight == transpose

  def matrix(self,mode,transpose=False):
    """The matrix associated with this mode - eg if mode is p(i,o) return
    a sparse matrix M_p so that v*M_p is appropriate for forward
    propagation steps from v.  If mode is p(o,i) then return the
    transpose of M_p.
    """
    assert mode.arity==2,'arity of '+str(mode) + ' is wrong: ' + str(mode.arity)
    assert (mode.functor,mode.arity) in self.matEncoding, \
           "can't find matrix for %s: is this defined in the program or database?" % str(mode)
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
    return a row vector equivalent to 1 * M_p^T."""
    return self.matrixPreimageOnes(mode) * self.matrixPreimageMat(mode)

  def matrixPreimageMat(self,mode):
    """Return the matrix M such that the preimage associated with
    this mode is ones*M """
    return self.matrix(mode,transpose=True)

  def matrixPreimageOnesType(self,mode):
    """Return the type t such that the preimage associated with
    this mode is db.ones(typeName=t)*M """
    functor = mode.getFunctor()
    if self.transposeNeeded(mode,transpose=True):
      return self.schema.getRange(functor,2)
    else:
      return self.schema.getDomain(functor,2)

  def matrixPreimageOnes(self,mode):
    """Return the ones vector v such that the preimage associated with
    this mode is v*M """
    return self.ones(self.matrixPreimageOnesType(mode))

  #
  # handling parameters
  #

  def isParameter(self,mode):
    return (mode.functor,mode.arity) in self.paramSet

  def markAsParam(self,functor,arity):
    logging.warn('MatrixDB.markAsParam is deprecated - use markAsParameter')
    self.markAsParameter(functor,arity)

  def markAsParameter(self,functor,arity):
    """ Mark a predicate as a parameter """
    if (functor,arity) not in self.paramSet:
      self.paramSet.add((functor,arity))
      self.paramList.append((functor,arity))

  def clearParameterMarkings(self):
    """ Clear previously marked parameters"""
    self.paramSet = set()
    self.paramList = []

  def getParameter(self,functor,arity):
    assert (functor,arity) in self.paramSet,'%s/%d not a parameter' % (functor,arity)
    return self.matEncoding[(functor,arity)]

  def parameterIsInitialized(self,functor,arity):
    return (functor,arity) in self.matEncoding

  def setParameter(self,functor,arity,replacement):
    assert (functor,arity) in self.paramSet,'%s/%d not a parameter' % (functor,arity)
    self.matEncoding[(functor,arity)] = replacement

  #
  # convert from vectors, matrixes to symbols - for i/o and debugging
  #

  def asSymbol(self,symbolId,typeName=None):
    """ Convert a typed integer id to a symbol
    """
    typeName = self._fillDefault(typeName)
    return self.schema.getSymbol(typeName,symbolId)

  def asSymbolId(self,symbol,typeName=None):
    """ Convert a typed symbol to an integer id
    """
    typeName = self._fillDefault(typeName)
    if self.schema.hasId(typeName,symbol):
      return self.schema.getId(typeName,symbol)
    else:
      return -1

  def rowAsSymbolDict(self,row,typeName=None):
    if typeName is None: typeName = THING
    result = {}
    coorow = row.tocoo()
    for i in range(len(coorow.data)):
      assert coorow.row[i]==0,"Expected 0 at coorow.row[%d]" % i
      s = self.schema.getSymbol(typeName,coorow.col[i])
      result[s] = coorow.data[i]
    return result

  def matrixAsSymbolDict(self,m,typeName=None):
    if typeName is None: typeName = THING
    result = {}
    (rows,cols)=m.shape
    for r in range(rows):
      result[r] = self.rowAsSymbolDict(m.getrow(r),typeName=typeName)
    return result

  def matrixAsPredicateFacts(self,functor,arity,m):
    result = {}
    m1 = scipy.sparse.coo_matrix(m)
    typeName1 = self.schema.getArgType(functor,arity,0)
    if arity==2:
      typeName2 = self.schema.getArgType(functor,arity,1)
      for i in range(len(m1.data)):
        a = self.schema.getSymbol(typeName1,m1.row[i])
        b = self.schema.getSymbol(typeName2,m1.col[i])
        w = m1.data[i]
        result[parser.Goal(functor,[a,b])] = w
    else:
      assert arity==1,"Arity (%d) must be 1 or 2" % arity
      for i in range(len(m1.data)):
        assert m1.row[i]==0, "Expected 0 at m1.row[%d]" % i
        b = self.schema.getSymbol(typeName1,m1.col[i])
        w = m1.data[i]
        if b==None:
          if i==0 and w<1e-10:
            logging.warn('ignoring low weight %g placed on index 0 for type %s in predicate %s' % (w,typeName1,functor))
          elif i==0:
            logging.warn('ignoring large weight %g placed on index 0 for type %s in predicate %s' % (w,typeName1,functor))
          else:
            assert False,'cannot find symbol on fact with weight %g for index %d for type %s in predicate %s' % (w,i,typeName1,functor)
        if b is not None:
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
    if not self.isTypeless():
      for (functor,arity),m in sorted(self.matEncoding.items()):
        typenames = map(lambda i:self.schema.getArgType(functor,arity,i), range(arity))
        print 'typing: %s(%s)' % (functor,",".join(typenames))

  def numMatrices(self):
    return len(self.matEncoding.keys())

  def size(self):
    return sum(map(lambda m:m.nnz, self.matEncoding.values()))

  def parameterSize(self):
    return sum([m.nnz for  ((fun,arity),m) in self.matEncoding.items() if (fun,arity) in self.paramSet])

  def createPartner(self):
    """Create a 'partner' datavase, which shares the same symbol table,
    but not the same data. Matrices/relations can be moved back
    and forth between partners.  Used mainly for testing."""
    partner = MatrixDB()
    partner.schema = self.schema
    return partner

  #
  # i/o
  #

  def serialize(self,direc):
    if not os.path.exists(direc):
      os.makedirs(direc)
    self.schema.serialize(direc)
    scipy.io.savemat(os.path.join(direc,"db.mat"),self.matEncoding,do_compression=True)

  @staticmethod
  def deserialize(direc):
    db = MatrixDB()
    db.schema = schema.AbstractSchema.deserialize(direc)
    scipy.io.loadmat(os.path.join(direc,"db.mat"),db.matEncoding)
    #serialization/deserialization ends up converting
    #(functor,arity) pairs to strings and csr_matrix to csc_matrix
    #so convert them back....
    for stringKey,mat in db.matEncoding.items():
      del db.matEncoding[stringKey]
      if not stringKey.startswith('__'):
        db.matEncoding[eval(stringKey)] = scipy.sparse.csr_matrix(mat)
    logging.info('deserialized database has %d relations and %d non-zeros' % (db.numMatrices(),db.size()))
    db.checkTyping()
    return db

  @staticmethod
  def uncache(dbFile,factFile):
    """Build a database file from a factFile, serialize it, and return
    the de-serialized database.  Or if that's not necessary, just
    deserialize it.  As always the factFile can be a
    colon-separated list.
    """
    if not os.path.exists(dbFile) or any([os.path.getmtime(f)>os.path.getmtime(dbFile) for f in factFile.split(":")]):
      logging.info('serializing fact file %s to %s' % (factFile,dbFile))
      db = MatrixDB.loadFile(factFile)
      db.serialize(dbFile)
      os.utime(dbFile,None) #update the modification time for the directory
      return db
    else:
      logging.info('deserializing db file '+ dbFile)
      return MatrixDB.deserialize(dbFile)

  # high level routines for loading files

  def addLines(self,lines):
    """ Clear the buffers, add lines, and flush the buffers.
    """
    self.startBuffers()
    for line in lines:
      self._bufferLine(line,'<no file>',0)
    self.flushBuffers()

  @staticmethod
  def loadFile(filenames):
    """Return a MatrixDB created by loading a file, or colon-separated
    list of files.
    """
    db = MatrixDB()
    db.startBuffers()
    for f in filenames.split(":"):
      db.bufferFile(f)
      logging.info('buffered file %s' % f)
    db.flushBuffers()
    logging.info('loaded database has %d relations and %d non-zeros' % (db.numMatrices(),db.size()))
    return db

  # manage buffers used to store matrix data before it is inserted

  def startBuffers(self):
    #buffer data for a sparse matrix: buf[pred][i][j] = f
    #TODO: would lists and a coo matrix make a nicer buffer?
    #def dictOfFloats(): return collections.defaultdict(float)
    #def dictOfFloatDicts(): return collections.defaultdict(dictOfFloats)
    #self._buf = collections.defaultdict(dictOfFloatDicts)
    self._databuf = collections.defaultdict(list)
    self._rowbuf = collections.defaultdict(list)
    self._colbuf = collections.defaultdict(list)

  def bufferFile(self,filename):
    """Load triples from a file and buffer them internally."""
    k = 0
    for line in open(filename):
      k += 1
      if not k%10000: logging.info('read %d lines' % k)
      self._bufferLine(line,filename,k)

  def flushBuffers(self):
    """Flush all triples from the buffer."""
    for f,arity in self._databuf.keys():
      self._flushBuffer(f,arity)
    self._databuf = None
    self.startBuffers()

  def _flushBuffer(self,functor,arity):
    """Flush the triples defining predicate p from the buffer and define
    p's matrix encoding"""
    key = (functor,arity)
    logging.info('flushing %d buffered non-zero values for predicate %s' % (len(self._databuf[key]),functor))
    if arity==2:
      nrows = self.schema.getMaxId(self.schema.getDomain(functor,arity)) + 1
      ncols = self.schema.getMaxId(self.schema.getRange(functor,arity)) + 1
    else:
      nrows = 1
      ncols = self.schema.getMaxId(self.schema.getDomain(functor,arity)) + 1
    coo_matrix = scipy.sparse.coo_matrix((self._databuf[key],(self._rowbuf[key],self._colbuf[key])), shape=(nrows,ncols))
    self.matEncoding[key] = scipy.sparse.csr_matrix(coo_matrix,dtype='float32')
    self.matEncoding[key].sort_indices()
    mutil.checkCSR(self.matEncoding[key], 'flushBuffer %s/%d' % key)

  def _bufferTriplet(self,functor,arity,a1,a2,w,filename,k):
    key = (functor,arity)
    if (key in self.matEncoding):
      logging.error("predicate encoding is already completed for "+str(key)+ " at line: "+line)
      return
    ti = self.schema.getArgType(functor,arity,0)
    tj = self.schema.getArgType(functor,arity,1)
    if ti is None or (tj is None and arity==2):
      logging.error('line %d of %s: undeclared relation %s/%d' % (k,filename,functor,arity))
    else:
      i = self.schema.getId(ti, a1)
      self._databuf[key].append(w)
      if arity==1:
        self._rowbuf[key].append(0)
        self._colbuf[key].append(i)
      else:
        assert arity==2 and a2 is not None
        self._rowbuf[key].append(i)
        j = self.schema.getId(tj, a2)
        self._colbuf[key].append(j)

  #
  # the real work in parsing a .cfacts file
  #

  def _bufferLine(self,line,filename,k):

    """Load a single triple encoded as a tab-separated line.."""
    def _atof(s):
      try:
        return float(s)
      except ValueError:
        return None

    line = line.strip()
    # blank lines
    if not line: return
    # declarations
    if line.startswith('#'):
      # look for a type declaration
      place = line.find(':-')
      if place>=0:
        decl = declare.TypeDeclaration(line[place+len(':-'):].strip())
        if decl.getFunctor()==TRAINABLE_DECLARATION_FUNCTOR and decl.getArity()==2 and (decl.arg(1) in ['1','2']):
          # declaration is trainable(foo,1) or trainable(foo,2)
          trainableFunctor = decl.arg(0)
          trainableArity = int(decl.arg(1))
          self.markAsParameter(trainableFunctor,trainableArity)
        else:
          # if possible, over-ride the default 'untyped' schema with one that can handle the type declaration
          if self.schema.isTypeless():
            if self.schema.empty() and not conf.ignore_types:
              self.schema = schema.TypedSchema()
          if not conf.ignore_types:
            self.schema.declarePredicateTypes(decl.functor,decl.args())
      return

    # data lines
    parts = line.split("\t")
    if len(parts)==4:
      # must be functor,a1,a2,weight
      functor,a1,a2,weight_string = parts[0],parts[1],parts[2],parts[3]
      w = _atof(weight_string)
      if w is None:
        logging.error('line %d of %s: illegal weight' % (k,filename,weight_string))
        return
      self._bufferTriplet(functor,2,a1,a2,w,filename,k)
    elif len(parts)==2:
      # must be functor,a1
      functor,a1 = parts[0],parts[1]
      self._bufferTriplet(functor,1,a1,None,1.0,filename,k)
    elif len(parts)==3:
      # might be functor,a1,a2 OR functor,a1,weight
      possible_weight_string = parts[2]
      w = _atof(possible_weight_string)
      if self.schema.isTypeless() and (w is not None) and conf.allow_weighted_tuples:
        functor,a1 = parts[0],parts[1]
        self._bufferTriplet(functor,1,a1,None,w,filename,k)
      elif self.schema.isTypeless():
        # can't make this a weighted tuple
        functor,a1,a2 = parts[0],parts[1],parts[2]
        self._bufferTriplet(functor,2,a1,a2,1.0,filename,k)
      elif not self.schema.isTypeless():
        functor = parts[0]
        if self.schema.getDomain(functor,2) and not self.schema.getDomain(functor,1):
          # must be binary
          a1,a2 = parts[1],parts[2]
          self._bufferTriplet(functor,2,a1,a2,1.0,filename,k)
        elif self.schema.getDomain(functor,1) and not self.schema.getDomain(functor,2):
          assert w is not None,'line %d file %s: illegal weight %s' % (k,filename,possible_weight_string)
          a1 = parts[1]
          self._bufferTriplet(functor,1,a1,None,1.0,filename,k)
        elif w is not None:
          a1 = parts[1]
          logging.warn('line %d file %s: assuming %s is a weight' % (k,filename,possible_weight_string))
          self._bufferTriplet(functor,1,a1,None,w,filename,k)
        else:
          a1,a2 = parts[1],parts[2]
          self._bufferTriplet(functor,2,a1,a2,1.0,filename,k)
    else:
      logging.error('line %d file %s: illegal line %r' % (k,filename,line))
      return
