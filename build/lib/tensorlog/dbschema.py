# (C) William W. Cohen and Carnegie Mellon University, 2017
#
# define a database schema
#
import os.path
import logging

from tensorlog import util

THING = '__THING__' # name of default type
NULL_ENTITY_NAME = '__NULL__'  #name of null entity marker
OOV_ENTITY_NAME = '__OOV__'  #name of out-of-vocabulary marker entity

class AbstractSchema(object):

  def checkTyping(self):
    """ Raise an error or print a warning if the types are not correct
    """
    assert False, 'abstract method called'

  def isTypeless(self):
    """ True if the database has no defined types
    """
    assert False, 'abstract method called'

  def defaultType(self):
    """ Return the name of the default type, if there is one.
    """

  def getTypes(self):
    """ Return a list of all defined types
    """
    assert False, 'abstract method called'

  def insertType(self,typeName):
    """ Add a new type
    """
    assert False, 'abstract method called'

  def getDomain(self,functor,arity):
    """ Return type that is the domain of a predicate """
    assert False, 'abstract method called'

  def getRange(self,functor,arity):
    """ Return type that is the range of a predicate """
    assert False, 'abstract method called'

  def getArgType(self,functor,arity,i):
    """Return the type associated with argument i of a predicate, where
    0<=i<arity.
    """
    assert False, 'abstract method called'

  def declarePredicateTypes(self,functor,types):
    """ Add a type declaration to the schema.
    """
    assert False, 'abstract method called'

  def serialize(self,direc):
    """ Save to files in a directory
    """
    assert False, 'abstract method called'

  @staticmethod
  def deserialize(direc):
    """ Restore from serialized files in a directory
    """
    symbolFile = os.path.join(direc,"symbols.txt")
    if os.path.isfile(symbolFile):
      return UntypedSchema.deserializeFrom(symbolFile)
    else:
      return TypedSchema.deserializeFrom(os.path.join(direc,"typed-symbols.txt"))

  def getMaxId(self,typeName):
    """ Return max id of any symbol for this type
    """
    assert False, 'abstract method called'

  def hasId(self,typeName,sym):
    """ Return true if this symbol has been added to this type
    """
    assert False, 'abstract method called'

  def getId(self,typeName,sym):
    """Return id for this symbol in the type, adding the symbol if
    necessary.
    """
    assert False, 'abstract method called'

  def getSymbol(self,typeName,symbolId):
    """Return string symbol for this id in the type, adding the symbol if
    necessary.
    """
    assert False, 'abstract method called'


  def _safeSymbTab(self):
    """ Symbol table with reserved words 'i', 'o', and 'any'
    """
    result = SymbolTable()
    result.reservedSymbols.add("i")
    result.reservedSymbols.add("o")
    result.reservedSymbols.add(THING)
    # always insert special entity names first
    result.insert(NULL_ENTITY_NAME)
    assert result.getId(NULL_ENTITY_NAME)==1
    result.insert(OOV_ENTITY_NAME)
    result._empty = True
    return result

  def _checkAndInsertSymbols(self,typeName,symbolFile):
    """ worker routine used by load methods
    """
    k = 1
    for line in open(symbolFile):
      sym = line.strip()
      i = self.getId(typeName,sym)
      assert i==k,'symbols out of sync for symbol "%s" type %d: expected index %d actual %d' % (sym,typeName,i,k)
      k += 1

class UntypedSchema(AbstractSchema):
  """ A trivial schema where everything is a default type
  """

  def __init__(self):
    # for consistency use the same repr as TypedSchema
    self._stab = { THING:self._safeSymbTab() }

  def checkTyping(self,predicateList):
    pass

  def isTypeless(self):
    return True

  def empty(self):
    return self._stab[THING]._empty

  def defaultType(self):
    return THING

  def getTypes(self):
    return [THING]

  def insertType(self,typeName):
    assert False,'inserting new type in untyped schema'

  def getDomain(self,functor,arity):
    return THING

  def getRange(self,functor,arity):
    return THING

  def getArgType(self,functor,arity,i):
    return THING

  def declarePredicateTypes(self,functor,types):
    assert False, 'predicate declared but database schema is untyped'

  def serialize(self,direc):
    """Save info needed to deserialize this object in appropriately named
    file in the given directory
    """
    with open(os.path.join(direc,'symbols.txt'), 'w') as fp:
      self.serializeTo(fp)

  def serializeTo(self,fpLike):
    """Serialize the info needed to deserialize this object in a stream -
    ie any object which supports the write method.
    """
    for i in range(1,self.getMaxId(THING)+1):
      fpLike.write(self.getSymbol(THING,i) + '\n')

  @staticmethod
  def deserializeFrom(fileLike):
    result = UntypedSchema()
    k = 1
    for line in util.linesIn(fileLike):
      sym = line.strip()
      i = result.getId(THING,sym)
      assert i==k,'symbols out of sync for symbol "%s": expected index %d actual %d' % (sym,i,k)
      k += 1
    return result

  def getMaxId(self,typeName):
    """ Return max id of any symbol for this type
    """
    return self._stab[THING].getMaxId()

  def hasId(self,typeName,sym):
    """ Return true if this symbol has been added to this type
    """
    return self._stab[THING].hasId(sym)

  def getId(self,typeName,sym):
    """Return id for this symbol in the type, adding the symbol if
    necessary.
    """
    return self._stab[THING].getId(sym)

  def getSymbol(self,typeName,symbolId):
    """Return string symbol for this id in the type, adding the symbol if
    necessary.
    """
    return self._stab[THING].getSymbol(symbolId)


class TypedSchema(AbstractSchema):

  def __init__(self):
    # self._type[i][(functor,arity)] is name of type for argument i of that predicate
    self._type = { 0:{}, 1:{} }
    self._stab = {}
    self._declarations = []

  def __str__(self):
    return 'DB Schema with %d declarations: %s' % (len(self._declarations), " & ".join(self._declarations))

  def checkTyping(self,predicateList,strict=False):
    """ Raise an error or print a warning if types of the functor,arity pairs in the
    list have not been declared:
    """
    for (functor,arity) in predicateList:
      for i in range(arity):
        if (i not in self._type) or ((functor,arity) not in self._type[i]):
          logging.warn('matrixDB relation %s/%d has no type declared for argument %d' % (functor,arity,i+1))
          if strict: assert False,'inconsistent use of types'
        else:
          typeName = self._type[i][(functor,arity)]
          logging.info('matrixDB relation %s/%d argument %d type %s' % (functor,arity,i+1,typeName))

  def isTypeless(self):
    return False

  def insertType(self,typeName):
    self._stab[typeName] = self._safeSymbTab()

  def defaultType(self):
    assert False, 'TypedSchema has no default type! you need to declare types for all predicates'

  def getTypes(self):
    return list(self._stab.keys())

  def getDomain(self,functor,arity):
    return self.getArgType(functor,arity,0)

  def getRange(self,functor,arity):
    return self.getArgType(functor,arity,1)

  def getArgType(self,functor,arity,i):
    return self._type[i].get((functor,arity))

  def declarePredicateTypes(self,functor,types):
    self._declarations.append( '%s(%s)' % (functor,",".join(types)))
    arity = len(types)
    for i,typeName in enumerate(types):
      self._declarePredicateArgType(functor,arity,i,typeName)

  def _declarePredicateArgType(self,functor,arity,i,typeName):
    key = (functor,arity)
    if key in self._type[i] and self._type[i][key]!=typeName:
      errorMsg = '%s/%d argument %d declared as both type %r and %r' \
                      % (functor,arity,i,typeName,self._type[i][key])
      assert False, errorMsg
    self._type[i][key] = typeName
    if typeName not in self._stab:
      self._stab[typeName] = self._safeSymbTab()

  def serialize(self,direc):
    """Save info needed to deserialize this object in appropriately named
    file in the given directory
    """
    with open(os.path.join(direc,'typed-symbols.txt'), 'w') as fp:
      self.serializeTo(fp)

  def serializeTo(self,fp):
    for decl in self._declarations:
      fp.write(decl + '\n')
    for i in range(2):
      for (functor,arity) in self._type[i]:
        fp.write('\t'.join([str(i),functor,str(arity),self._type[i][(functor,arity)]]) + '\n')
    for typeName in self.getTypes():
      fp.write('\n' + typeName + '\n')
      for i in range(1,self.getMaxId(typeName)+1):
        fp.write(self.getSymbol(typeName,i) + '\n')

  @staticmethod
  def deserializeFrom(fileLike):
    result = TypedSchema()
    readingTypeDecs = True
    currentType = None
    k = -1
    for line in util.linesIn(fileLike):
      line = line.strip()
      if readingTypeDecs and line:
        # type declarations start out the file - either strings describing a declaration,
        # or else i,function,arity,type_of_arg_i_of_pred_defined_by_functor_and_arity
        parts = line.split("\t")
        if len(parts)==1:
          result._declarations.append(parts[0])
        else:
          iStr,functor,arityStr,typeName = parts
          result._declarePredicateArgType(functor,int(arityStr),int(iStr),typeName)
      elif readingTypeDecs and not line:
        # empty line terminates type declarations
        readingTypeDecs = False
        currentType = None
      elif not readingTypeDecs and line and currentType is None:
        # first line after empty line (signalled by 'currentType is None') is type name
        currentType = line
        result.insertType(currentType)
        k = 1
      elif not readingTypeDecs and line and currentType is not None:
        # lines following the name of a type are symbols for that type
        sym = line
        i = result.getId(currentType,sym)
        assert i==k,'symbols out of sync for symbol "%s" type %d: expected index %d actual %d' % (sym,currentType,i,k)
        k += 1
      elif not readingTypeDecs and not line:
        # empty line terminates list of symbols for a type
        currentType = None
      else:
        assert False,'cannot deserialize a TypedSchema from %r' % fileLike
    return result

  def getMaxId(self,typeName):
    """ Return max id of any symbol for this type
    """
    return self._stab[typeName].getMaxId()

  def hasId(self,typeName,sym):
    """ Return true if this symbol has been added to this type
    """
    return self._stab[typeName].hasId(sym)

  def getId(self,typeName,sym):
    """Return id for this symbol in the type, adding the symbol if
    necessary.
    """
    return self._stab[typeName].getId(sym)

  def getSymbol(self,typeName,symbolId):
    """Return string symbol for this id in the type, adding the symbol if
    necessary.
    """
    return self._stab[typeName].getSymbol(symbolId)


#TODO: do I need reserved symbols? index to start at 0?

class SymbolTable(object):
  """A symbol table mapping strings to/from integers in the range 1..N
  inclusive."""

  def __init__(self,initSymbols=[]):
    self.reservedSymbols = set()
    self._symbolList = [None]
    self._nextId = 0
    self._idDict = {}
    for s in initSymbols:
      self.insert(s)
    self._empty = True

  def insert(self,symbol):
    """Insert a symbol."""
    if symbol not in self._idDict:
      self._nextId += 1
      self._idDict[symbol] = self._nextId
      self._symbolList += [symbol]
      self._empty = False

  def getSymbolList(self):
    """Get an array of all defined symbols."""
    return self._symbolList[1:]

  def getSymbol(self,id):
    return self._symbolList[id]

  def hasId(self,symbol):
    return symbol in self._idDict

  def getId(self,symbol):
    """Get the numeric id, between 1 and N, of a symbol.
    """
    self.insert(symbol)
    return self._idDict[symbol]

  def getMaxId(self):
    return self._nextId
