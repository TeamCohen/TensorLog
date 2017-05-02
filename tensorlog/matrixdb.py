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
from tensorlog import declare
from tensorlog import symtab
from tensorlog import parser
from tensorlog import mutil

conf = config.Config()
conf.allow_weighted_tuples = True; conf.help.allow_weighted_tuples = 'Allow last column of cfacts file to be a weight for the fact'
conf.ignore_types = False;         conf.help.ignore_types = 'Ignore type declarations'

#name of null entity, which is returned when a proof fails
NULL_ENTITY_NAME = '__NULL__'
#name of out-of-vocabulary marker entity
OOV_ENTITY_NAME = '__OOV__'
#name of default type
THING = '__THING__'
#functor in declarations of trainable relations, eg trainable(posWeight,1)
TRAINABLE_DECLARATION_FUNCTOR = 'trainable'

class MatrixDB(object):
  """ A logical database implemented with sparse matrices """

  def __init__(self):
    #maps symbols to numeric ids
    self._stab = {THING: self._safeSymbTab() }
    #matEncoding[(functor,arity)] encodes predicate as a matrix
    self.matEncoding = {}
    # mark which matrices are 'parameters' by (functor,arity) pair
    self.paramSet = set()
    self.paramList = []
    # buffer initialization: see startBuffers()
    self._buf = None
    # type information - indexed by (functor,arity) pair
    # defaulting to 'THING'
    self._type = collections.defaultdict( lambda:collections.defaultdict(lambda:THING) )

  def _safeSymbTab(self):
    """ Symbol table with reserved words 'i', 'o', and 'any'
    """
    result = symtab.SymbolTable()
    result.reservedSymbols.add("i")
    result.reservedSymbols.add("o")
    result.reservedSymbols.add(THING)
    # always insert special entity names first
    result.insert(NULL_ENTITY_NAME)
    assert result.getId(NULL_ENTITY_NAME)==1
    result.insert(OOV_ENTITY_NAME)
    return result

  def checkTyping(self,strict=False):
    if self.isTypeless():
      logging.info('untyped matrixDB passed checkTyping')
    else:
      for i,d in enumerate(self._type.values()):
        for (functor,arity),typeName in d.items():
          if typeName==THING:
            logging.warn('matrixDB relation %s/%d has no type declared for argument %d' % (functor,arity,i+1))
            if strict: assert False,'inconsistent use of types'
          else:
            logging.info('matrixDB relation %s/%d argument %d type %s' % (functor,arity,i+1,typeName))

  def isTypeless(self):
    return len(self._stab.keys())==1

  def declaredType(self,functor,arity):
    return (functor,arity) in self._type[0]

  def getTypes(self):
    return self._stab.keys()

  def getDomain(self,functor,arity,frozen=False):
    """ Domain of a predicate """
    return self.getArgType(functor,arity,0)

  def getRange(self,functor,arity,frozen=False):
    """ Range of a predicate """
    return self.getArgType(functor,arity,1)

  def getArgType(self,functor,arity,i,frozen=False):
    """ Type associated with argument i of a predicate"""
    if not frozen:
      return self._type[i][(functor,arity)]
    else:
      return self._type[i].get((functor,arity),THING)

  def addTypeDeclaration(self,decl,filename,lineno):
    if conf.ignore_types:
      logging.info('ignoring type declaration %s at %s:%d' % (str(decl),filename,lineno))
    else:
      logging.info('type declaration %s at %s:%d' % (str(decl),filename,lineno))
      key = (decl.functor,decl.arity)
      for j in range(decl.arity):
        typeName = decl.getType(j)
        if key in self._type[j] and self._type[j][key] != typeName:
          errorMsg = '%s:%d:  %s/%d argument %d declared as both type %s and %s' \
                      % (filename,lineno,decl.functor,decl.arity,j,typeName,self._type[j][key])
          assert False, errorMsg
        else:
          if typeName not in self._stab:
            self._stab[typeName] = self._safeSymbTab()
          self._type[j][key] = typeName

  #
  # retrieve matrixes, vectors, etc
  #

  def _fillDefault(self,typeName):
    if typeName is None or conf.ignore_types: return THING
    else: return typeName

  def dim(self,typeName=None):
    typeName = self._fillDefault(typeName)
    """Number of constants in the database, and dimension of all the vectors/matrices."""
    return self._stab[typeName].getMaxId() + 1

  def onehot(self,s,typeName=None,outOfVocabularySymbolsAllowed=False):
    typeName = self._fillDefault(typeName)
    """A onehot row representation of a symbol."""
    if outOfVocabularySymbolsAllowed and not self._stab[typeName].hasId(s):
        return self.onehot(OOV_ENTITY_NAME,typeName)
    assert self._stab[typeName].hasId(s),'constant %s (type %s) not in db' % (s,typeName)
    n = self.dim(typeName)
    i = self._stab[typeName].getId(s)
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
      return self.getRange(functor,2)
    else:
      return self.getDomain(functor,2)

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
    if typeName is None: typeName = THING
    return self._stab[typeName].getSymbol(symbolId)

  def asSymbolId(self,symbol,typeName=None):
    """ Convert a typed symbol to an integer id
    """
    if typeName is None: typeName = THING
    stab = self._stab[typeName]
    if stab.hasId(symbol):
      return stab.getId(symbol)
    else:
      return -1

  def rowAsSymbolDict(self,row,typeName=None):
    if typeName is None: typeName = THING
    result = {}
    coorow = row.tocoo()
    for i in range(len(coorow.data)):
      assert coorow.row[i]==0,"Expected 0 at coorow.row[%d]" % i
      s = self._stab[typeName].getSymbol(coorow.col[i])
      result[s] = coorow.data[i]
    return result

  def arrayAsSymbolDict(self,arr,typeName=None):
    if typeName is None: typeName = THING
    result = {}
    for i in range(len(arr)):
            s = self._stab[typeName].getSymbol(i)
            result[s] = arr[i]
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
    typeName1 = self.getArgType(functor,arity,0)
    if arity==2:
      typeName2 = self.getArgType(functor,arity,1)
      for i in range(len(m1.data)):
        a = self._stab[typeName1].getSymbol(m1.row[i])
        b = self._stab[typeName2].getSymbol(m1.col[i])
        w = m1.data[i]
        result[parser.Goal(functor,[a,b])] = w
    else:
      assert arity==1,"Arity (%d) must be 1 or 2" % arity
      for i in range(len(m1.data)):
        assert m1.row[i]==0, "Expected 0 at m1.row[%d]" % i
        b = self._stab[typeName1].getSymbol(m1.col[i])
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
        typenames = map(lambda i:self.getArgType(functor,arity,i,frozen=True), range(arity))
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
    partner._stab = self._stab
    return partner

  #
  # i/o
  #

  def serialize(self,direc):
    if not os.path.exists(direc):
      os.makedirs(direc)
    if self.isTypeless():
      # old format - one symbol table
      with open(os.path.join(direc,"symbols.txt"), 'w') as fp:
        for i in range(1,self.dim(THING)):
          fp.write(self._stab[THING].getSymbol(i) + '\n')
    else:
      # write relation type information
      with open(os.path.join(direc,'types.txt'),'w') as fp:
        for i in range(2):
          for (functor,arity) in self._type[i]:
            fp.write('\t'.join([str(i),functor,str(arity),self._type[i][(functor,arity)]]) + '\n')
      # write each symbol table
      for typeName in self._stab.keys():
        with open(os.path.join(direc,typeName+"-symbols.txt"), 'w') as fp:
          for i in range(1,self.dim(typeName)):
            fp.write(self._stab[typeName].getSymbol(i) + '\n')
    scipy.io.savemat(os.path.join(direc,"db.mat"),self.matEncoding,do_compression=True)

  @staticmethod
  def deserialize(direc):
    db = MatrixDB()
    def checkSymbols(typeName,symbolFile):
      k = 1
      for line in open(symbolFile):
        sym = line.strip()
        i = db._stab[typeName].getId(sym)
        assert i==k,'symbols out of sync for symbol "{sym}" type {typ}: expected index {index} actual {actual}'.format(sym=sym,typ=typeName,index=i,actual=k)
        k += 1
    symbolFile = os.path.join(direc,"symbols.txt")
    if os.path.isfile(symbolFile):
      checkSymbols(THING,symbolFile)
    else:
      # read relation type information
      for line in open(os.path.join(direc,'types.txt')):
        iStr,functor,arityStr,typeStr = line.strip().split("\t")
        db._type[int(iStr)][(functor,int(arityStr))] = typeStr
      # read each symbol table
      for f0 in os.listdir(direc):
        f = os.path.join(direc,f0)
        if os.path.isfile(f) and str(f).endswith("-symbols.txt"):
          typeName = str(f0)[:-len("-symbols.txt")]
          if typeName not in db._stab:
            db._stab[typeName] = db._safeSymbTab()
          checkSymbols(typeName,f)
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

  def _bufferLine(self,line,filename,k):
    """Load a single triple encoded as a tab-separated line.."""
    def atof(s):
      try:
        return float(s)
      except ValueError:
        return float(0.0)

    line = line.strip()

    if not line: return
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
          self.addTypeDeclaration(decl,filename,k)
      return

    # buffer the parts of the line, which can be have either 1 or 2
    # arguments and optionally a numeric weight
    parts = line.split("\t")
    if conf.allow_weighted_tuples and len(parts)==4:
      functor,a1,a2,wstr = parts[0],parts[1],parts[2],parts[3]
      arity = 2
      w = atof(wstr)
    elif len(parts)==3:
      functor,a1,a2 = parts[0],parts[1],parts[2]
      w = atof(a2)
      if not conf.allow_weighted_tuples or w==0:
        arity = 2
        w = float(1.0)
      else:
        arity = 1
    elif len(parts)==2:
      functor,a1,a2 = parts[0],parts[1],None
      arity = 1
      w = float(1.0)
    else:
      logging.error("bad line '"+line+" '" + repr(parts)+"'")
      return
    key = (functor,arity)
    if (key in self.matEncoding):
      logging.error("predicate encoding is already completed for "+str(key)+ " at line: "+line)
      return
    i = self._stab[self.getArgType(functor,arity,0)].getId(a1)
    if not a2:
      j = -1
    else:
      j = self._stab[self.getArgType(functor,arity,1)].getId(a2)
    self._buf[key][i][j] = w

  def bufferFile(self,filename):
    """Load triples from a file and buffer them internally."""
    k = 0
    for line in open(filename):
      k += 1
      if not k%10000: logging.info('read %d lines' % k)
      self._bufferLine(line,filename,k)

  def flushBuffers(self):
    """Flush all triples from the buffer."""
    for f,arity in self._buf.keys():
      self.flushBuffer(f,arity)

  def flushBuffer(self,f,arity):
    """Flush the triples defining predicate p from the buffer and define
    p's matrix encoding"""
    logging.info('flushing %d buffered rows for predicate %s' % (len(self._buf[(f,arity)]),f))

    if arity==2:
      nrows = self._stab[self.getDomain(f,arity)].getMaxId() + 1
      ncols = self._stab[self.getRange(f,arity)].getMaxId() + 1
      m = scipy.sparse.lil_matrix((nrows,ncols),dtype='float32')
      for i in self._buf[(f,arity)]:
        for j in self._buf[(f,arity)][i]:
          m[i,j] = self._buf[(f,arity)][i][j]
      del self._buf[(f,arity)]
      self.matEncoding[(f,arity)] = scipy.sparse.csr_matrix(m,dtype='float32')
      self.matEncoding[(f,arity)].sort_indices()
    elif arity==1:
      ncols = self._stab[self.getDomain(f,arity)].getMaxId() + 1
      m = scipy.sparse.lil_matrix((1,ncols))
      for i in self._buf[(f,arity)]:
        for j in self._buf[(f,arity)][i]:
          m[0,i] = self._buf[(f,arity)][i][j]
      del self._buf[(f,arity)]
      self.matEncoding[(f,arity)] = scipy.sparse.csr_matrix(m,dtype='float32')
      self.matEncoding[(f,arity)].sort_indices()
    mutil.checkCSR(self.matEncoding[(f,arity)], 'flushBuffer %s/%d' % (f,arity))

  def rebufferMatrices(self):
    """Re-encode previously frozen matrices after a symbol table update"""
    for (functor,arity),m in self.matEncoding.items():
      targetNrows = self._stab[self.getDomain(functor,arity)].getMaxId() + 1
      targetNcols = self._stab[self.getRange(functor,arity)].getMaxId() + 1
      (rows,cols) = m.get_shape()
      if cols != targetNcols or rows != targetNrows:
        logging.info("Re-encoding predicate %s" % functor)
        if arity==2:
          # first shim the extra rows
          shim = scipy.sparse.lil_matrix((targetNrows-rows,cols))
          m = scipy.sparse.vstack([m,shim])
          (rows,cols) = m.get_shape()
        # shim extra columns
        shim = scipy.sparse.lil_matrix((rows,targetNcols-cols))
        self.matEncoding[(functor,arity)] = scipy.sparse.hstack([m,shim],format="csr")
        self.matEncoding[(functor,arity)].sort_indices()

  def clearBuffers(self):
    """Save space by removing buffers"""
    self._buf = None

  def startBuffers(self):
    #buffer data for a sparse matrix: buf[pred][i][j] = f
    #TODO: would lists and a coo matrix make a nicer buffer?
    def dictOfFloats(): return collections.defaultdict(float)
    def dictOfFloatDicts(): return collections.defaultdict(dictOfFloats)
    self._buf = collections.defaultdict(dictOfFloatDicts)

  def addLines(self,lines):
    self.startBuffers()
    for line in lines:
      self._bufferLine(line,'<no file>',0)
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
    colon-separated list of files.
    """
    db = MatrixDB()
    for f in filenames.split(":"):
      db.addFile(f)
    logging.info('loaded database has %d relations and %d non-zeros' % (db.numMatrices(),db.size()))
    return db

#
# test main
# --s serialize foo.cfacts foo.db
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
