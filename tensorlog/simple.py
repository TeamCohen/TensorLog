import collections
import logging
import os.path
import getopt
import sys
import time
import re

from tensorlog import bpcompiler
from tensorlog import comline
from tensorlog import declare
from tensorlog import dataset
from tensorlog import dbschema
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import parser
from tensorlog import program
from tensorlog import xctargets
if xctargets.tf:
  import tensorflow as tf
  from tensorlog import tensorflowxcomp
if xctargets.theano:
  from tensorlog import theanoxcomp



class Compiler(object):

  def __init__(self,target='tensorflow',db=None,prog=None,rule_features=True,autoset_db_params=True,summary_file=None):

    """Create an object with a simple interface that wraps a tensorlog compiler.
    Args:

      target: a string indicating the target language, currently
      'tensorflow' or 'theano'

      db: specifies the database used by tensorflow. Either a
        tensorlog.matrixdb.MatrixDB object, or a string that can be
        converted to one by tensorlog.comline.parseDBSpec.  The common
        cases of the latter are (a) a serialized tensorlog database,
        usually with extension .db or (b) a colon-separated list of
        files containing facts and type declarations (one per line).
        Facts are tab-separated and are of the form
        "binary_relation_name TAB head TAB tail [TAB weight]" or
        "unary_relation_name TAB head".  Type declarations are of one
        of these forms:

         # :- binary_relation_name(type_name1,type_name2)
         # :- unary_relation_name(type_name)

        where type_names are identifiers, which denote disjoint sets
        of DB entities.  Fact files usually have extension .cfacts.

        A db string can also be of the form "foo.db|bar.cfacts" in
        which case the serialized database foo.db will be used if it
        exists, and otherwise bar.cfacts will be loaded, parsed, and
        serialized in foo.db for later.

      prog: specifies a tensorlog program.  Either a
        tensorlog.program.Program object or a string that can be
        converted to one by tensorlog.comline.parseProgSpec, which
        currently would be a single filename.

      rule_features: if True, then the loaded program contains control
        features {...} on every rule (i.e., it will be
        tensorlog.program.ProPPRProgram object).

      autoset_db_params: if True, try and set parameter values
        automatically. This usually works for rule_features but not
        others.

      summary_file: if not None, and if target=='tensorflow', this
        location will be used as to hold summary data for tensorboard
        on the tensorlog operations.
    """

    # parse the db argument
    if isinstance(db,matrixdb.MatrixDB):
      self.db = db
    elif isinstance(db,DBWrapper):
      self.db = db.inner_db
    elif isinstance(db,str):
      self.db = comline.parseDBSpec(db)
    else:
      assert False,'cannot convert %r to a database' % db
    self.db.flushBuffers() # needed for DBWrappers

    # parse the program argument
    if prog is None:
      self.prog = program.ProPPRProgram(db=self.db, rules=RuleCollectionWrapper())
    elif isinstance(prog,program.Program):
      self.prog = prog
    elif isinstance(prog,Builder):
      self.prog = program.ProPPRProgram(db=self.db, rules=prog.rules)
    elif isinstance(prog,parser.RuleCollection):
      self.prog = program.ProPPRProgram(db=self.db, rules=prog)
    elif isinstance(prog,str):
      self.prog = comline.parseProgSpec(prog,self.db,proppr=rule_features)
    else:
      assert False,'cannot convert %r to a program' % prog

    # set weights
    if autoset_db_params:
      self.prog.setAllWeights()

    # parse the target argument
    self.target = target
    if target=='tensorflow':
      if not xctargets.tf: assert False, "tensorflow not available"
      self.xc = tensorflowxcomp.SparseMatDenseMsgCrossCompiler(self.prog, summaryFile=summary_file)
    elif target=='theano':
      if not xctargets.theano: assert False, "theano not available"
      self.xc = theanoxcomp.SparseMatDenseMsgCrossCompiler(self.prog)
    else:
      assert False,'illegal target %r: valid targets are "tensorflow" and "theano"' % target

  def get_cross_compiler(self):
    return self.xc

  def get_program(self):
    return self.prog

  def get_database(self):
    return self.db

  def proof_count(self,mode,inputs=None):
    """ An expression for the inference associated with a mode
    """
    args,expr = self.xc.proofCount(declare.asMode(mode),inputs=inputs)
    return expr

  def inference(self,mode,inputs=None):
    """ An expression for the inference associated with a mode
    """
    args,expr = self.xc.inference(declare.asMode(mode),inputs=inputs)
    return expr

  def loss(self,mode,inputs=None):
    """ An expression for the unregularized loss associated with a mode
    """
    args,expr = self.xc.dataLoss(declare.asMode(mode),inputs=inputs)
    return expr

  def trainable_db_variables(self,mode,for_optimization=False):
    """Return a list of expressions associated with predicates marked as
    parameters/trainable in the tensorlog database.  If
    for_optimization==True then return the underlying variables that
    are optimized, otherwise return expressions computing values that
    correspond most closely to the parameters.

    Eg, if a weight vector V is reparameterized by passing it through
    an softplus, so V=softplus(V0) is used in the proof_count
    expression, then for_optimization==True will return V0, and
    for_optimization==False will return V.
    """
    if for_optimization:
      return self.xc.getParamVariables(declare.asMode(mode))
    else:
      return self.xc.getParamHandles(declare.asMode(mode))

  #
  # needed for building feed_dicts for training/testing tensorflow
  #
  # note - you could also get the input placeholder from the args
  # returned by xc.inference and the output placeholder from
  # xcomp.TRAINING_TARGET_VARNAME
  #

  def input_placeholder_name(self,mode):
    """ For tensorflow, the name of the placeholder associated with the input to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getInputName(declare.asMode(mode))

  def input_placeholder(self,mode):
    """ For tensorflow, the placeholder associated with the input to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getInputPlaceholder(declare.asMode(mode))

  def target_output_placeholder_name(self,mode):
    """ For tensorflow, the name of the placeholder associated with the output to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getTargetOutputName(declare.asMode(mode))

  def target_output_placeholder(self,mode):
    """ For tensorflow, the placeholder associated with the output to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getTargetOutputPlaceholder(declare.asMode(mode))

  #
  # needed if you don't want to autoset the parameters stored in tensorlog's db
  #

  def db_param_list(self):
    """ Identifiers for trainable tensorlog DB relations. """
    return self.prog.getParamList()

  def db_param_is_set(self,param_id):
    """ Test to see if a parameter relation has a value. """
    (functor,arity) = param_id
    return self.db.parameterIsInitialized(functor,arity)

  def get_db_param_value(self,param_id):
    """ Get the value of a parameter relation has a value. """
    (functor,arity) = param_id
    return self.db.getParameter(functor,arity)

  def set_db_param_value(self,param_id,value):
    """Set the value of a parameter relation.  You can only usefully set a
    param BEFORE you start doing inference or training. This is
    because the value is stored in the tensorlog database first, then,
    when an inference or loss function is generated, the value will be
    used as the initializer for a variable.
    """
    (functor,arity) = param_id
    assert self.xc.parameterFromDBToExpr(functor,arity) is None,'too late to reset value for %r - it has already been used in the compiler'
    self.db.setParameter(functor,arity,value)

  #
  # useful after learning or for checkpointing
  #

  def set_all_db_params_to_learned_values(self,session):
    """ Set all trainable parameter relations to their learned values
    """
    self.xc.exportAllLearnedParams(session=session)

  def serialize_db(self,dir_name):
    """Save a serialized, quick-to-load version of the database."""
    self.db.serialize(dir_name)

  #
  # expose other useful routines
  #

  def _as_dataset(self,spec):
    if type(spec)==str:
      return comline.parseDatasetSpec(spec,self.db)
    elif '__iter__' in dir(spec):
      return dataset.Dataset.loadExamples(self.db,spec)
    else:
      assert False,'cannot load dataset from %r' % spec

  def load_dataset(self,dataset_spec):
    """Same as load_small_dataset, for backwards compatibility. """
    return self.load_small_dataset(dataset_spec)

  def load_small_dataset(self,dataset_spec):
    """Return a dictionary where keys are strings defining tensorlog
    functions - e.g., answer/io - and the values are pairs (X,Y) where
    X is a matrix that can be used as a batch input to the inference
    function, and Y is a matrix that is the desired output.

    Note that X is 'unwrapped', which may make it much larger than the
    sparse matrix which is stored.  If this exceeds memory Python
    usually just crashes.  In this case you should use
    load_big_dataset instead.

    Args:

      dataset_spec: a string specifying a tensorlog.dataset.Dataset.
        Usually this is either (a) a serialized dataset, with
        extension .dset or (b) a file with extension .exam, containing
        one example per line.  Each line is tab-separated and contains
        a predicate name p (which is assumed to have mode io); an
        input x to that predicate; a list of all correct outputs y,
        ie, the remaining tab-separated items are strings y such that
        p(x,y) should be true.

        A dataset_spec string can also be of the form
        "foo.dset|bar.exam" in which case the serialized dataset
        foo.dset will be used if it exists, and otherwise bar.exam
        will be loaded, parsed, and serialized in foo.dset for later.
    """
    dset = self._as_dataset(dataset_spec)
    return self.annotate_small_dataset(dset)
  def annotate_small_dataset(self,dset):
    m = dset.modesToLearn()[0]
    # convert to something bereft of tensorlog data structures: a
    # dictionary mapping strings like "p/io" to X,Y pairs, where X and
    # Y are wrapped inputs.
    def wrapped_xy_pair(mode): return (self.xc.wrapInput(dset.getX(mode)), self.xc.wrapInput(dset.getY(mode)))
    return dict((str(mode),wrapped_xy_pair(mode)) for mode in dset.modesToLearn())

  def modes_to_learn(self,dataset_obj):
    """List the functions that can be learned from this dataset_obj,
    where dataset_obj is returned from load_small_dataset or
    load_big_dataset.
    """
    if isinstance(dataset_obj,dict):
      return dataset_obj.keys()
    elif isinstance(dataset_obj,dataset.Dataset):
      return dataset_obj.modesToLearn()
    else:
      assert False,'illegal dataset object %r' % dataset_obj

  def minibatches(self,dataset_obj,batch_size=100,shuffle_first=True):
    """Yields a series of pairs (mode,(X,Y)) where X and Y are a minibatch
    suitable for training the function designated by mode.  Input is
    something returned by load_small_dataset or load_big_dataset.
    """
    if isinstance(dataset_obj,dict):
      dataset_dict = dataset_obj
      x_dict = {}
      y_dict = {}
      for mode_str,(x,y) in dataset_dict.items():
        mode = declare.asMode(mode_str)
        x_dict[mode] = self.xc.unwrapInput(x)
        y_dict[mode] = self.xc.unwrapInput(y)
        dset = dataset.Dataset(x_dict,y_dict)
        for mode,bx,by in dset.minibatchIterator(batchSize=batch_size,shuffleFirst=shuffle_first):
          yield str(mode),(self.xc.wrapInput(bx),self.xc.wrapInput(by))
    elif isinstance(dataset_obj, dataset.Dataset):
      dset = dataset_obj
      for mode,bx,by in dset.minibatchIterator(batchSize=batch_size,shuffleFirst=shuffle_first):
        yield str(mode),(self.xc.wrapInput(bx),self.xc.wrapInput(by))
    else:
      assert False,'illegal dataset object %r' % dataset_obj

  def load_big_dataset(self,dataset_spec,verbose=True):
    """Return a dataset object, which can be used as the first argument to
    tlog.minibatches to cycle through the examples.  The object is an
    instance of tensorlog.dataset.Dataset.

    Args:

      dataset_spec: a string specifying a tensorlog.dataset.Dataset.
    See documents for load_small_dataset.
    """

    dset = self._as_dataset(dataset_spec)
    return self.annotate_big_dataset(dset)
  def annotate_big_dataset(self,dset):
    for m in dset.modesToLearn():
      x = dset.getX(m)
      y = dset.getY(m)
      (rx,cx) = x.shape
      (ry,cy) = y.shape
      def rows_per_gigabyte(c): return (1024.0*1024.0*1024.0) / (c*4.0)
      sm = str(m)
      print 'mode %s: X is sparse %d x %d matrix (about %.1f rows/Gb)' % (sm,rx,cx,rows_per_gigabyte(cx))
      print 'mode %s: Y is sparse %d x %d matrix (about %.1f rows/Gb)' % (sm,ry,cy,rows_per_gigabyte(cy))
    return dset

class Builder(object):
  """Supports construction of programs within python, using the
  following sort of syntax.

    b = Builder()
    X,Y,Z = b.variables("X Y Z")
    aunt,parent,sister,wife = b.predicates("aunt parent sister wife")
    uncle = b.predicate("uncle")

    b.rules += aunt(X,Y) <= parent(X,Z),sister(Z,Y)
    b.rules += aunt(X,Y) <= uncle(X,Z),wife(Z,Y)

  Or, with 'control features'

    r1,r2 = b.rule_ids("r1 r2")
    b.rules += aunt(X,Y) <= uncle(X,Z) & wife(Z,Y) // r1
    b.rules += aunt(X,Y) <= parent(X,Z) & sister(Z,Y) // r2

  or

    b.rules += aunt(X,Y) <= uncle(X,Z) & wife(Z,Y) // (weight(F) | description(X,D) & feature(X,F))

  You can also construct a database schema

    person_t,place_t = b.types("person_t place_t") # note: singleton 'type' isn't available
    b.schema += aunt(person_t,person_t) & uncle(person_t,person_t)
    ...

  Or specify the database (either once you've established a schema, or
  else if the schema information is inline, of the database is
  untyped) as:

    b.db = "foo.db|f1.cfacts:f2.cfacts"

  or

    b.db = "foo.db"

  or

    b.db += "f1.cfacts"
    b.db += "f2.cfacts"
    ...

  """

  def __init__(self):
    self.rules = RuleCollectionWrapper()
    self.schema = SchemaWrapper(self)
    self.db = DBWrapper(self)

  def __setattr__(self,name,value):
    # this overrides b.db = ....
    if name=='db':
      if 'db' not in self.__dict__:
        # the assignment in Builder.__init__
        self.__dict__['db'] = value
      else:
        self.__dict__['db']._set_to_value(value)
    else:
      self.__dict__[name] = value

  @staticmethod
  def _split(comma_or_space_sep_names):
    return re.split("\\W+", comma_or_space_sep_names)

  @staticmethod
  def variable(variable_name):
    return Builder.variables(variable_name)[0]

  @staticmethod
  def variables(space_sep_variable_names):
    return Builder._split(space_sep_variable_names)

  @staticmethod
  def types(space_sep_type_names):
    return Builder._split(space_sep_type_names)

  @staticmethod
  def rule_id(type_name,rule_id):
    return Builder.rule_ids(type_name,rule_id)[0]

  @staticmethod
  def rule_ids(type_name,space_sep_rule_ids):
    def goal_builder(rule_id):
      var_name = rule_id[0].upper()+rule_id[1:]
      return RuleWrapper(
          None,
          [],
          features=[parser.Goal('weight',[var_name])],
          findall=[parser.Goal(bpcompiler.ASSIGN,[var_name,rule_id,type_name])])
    return map(goal_builder, Builder._split(space_sep_rule_ids))

  @staticmethod
  def predicate(predicate_name):
    return Builder.predicates(predicate_name)[0]

  @staticmethod
  def predicates(space_sep_predicate_names):
    def goal_builder(pred_name):
      def builder(*args):
        return RuleWrapper(None,[parser.Goal(pred_name,args)])
      return builder
    return map(goal_builder, Builder._split(space_sep_predicate_names))

  def __iadd__(self,other):
    if isinstance(other,parser.Rule):
      self.rules.add(other)
    else:
      assert False, 'rule syntax error for builder: %s += %s' % (str(self),str(other))
    return self

class RuleCollectionWrapper(parser.RuleCollection):
  """ Subclass RuleCollection to handle the += notation
  """
  def __init__(self):
    super(RuleCollectionWrapper,self).__init__(syntax='pythonic')

  def __iadd__(self,other):
    if isinstance(other,parser.Rule):
      self.add(other)
    else:
      assert False, 'rule syntax error for builder: %s += %s' % (str(self),str(other))
    return self

class DBWrapper(object):
  """ Wraps a database and supports builder.db = matrixDB, builder.db = db_spec
  where db_spec is parseable by comline, or builder.db += filename.
  """

  def __init__(self,builder):
    self.builder = builder
    self.inner_db = None

  def _set_to_value(self,other):
    init_schema = None if self.builder.schema.empty else self.builder.schema
    if isinstance(other,matrixdb.MatrixDB):
      if init_schema is not None:
        logging.warn('schema definition is discarded when you assign a database to builder.db')
      self.inner_db = other
    elif isinstance(other,str) and comline.isUncachefromSrc(other):
      cache,src = comline.getCacheSrcPair(other)
      self.inner_db = matrixdb.MatrixDB.uncache(cache,src,initSchema=init_schema)
    elif isinstance(other,str) and other.endswith(".db"):
      self.inner_db = matrixdb.MatrixDB.deserialize(other)
    elif isinstance(other,str):
      self.inner_db = matrixdb.MatrixDB.loadFile(other,initSchema=init_schema)
    elif isinstance(other,DBWrapper) and other is self:
      pass
    else:
      assert False,'builder.db can only be assigned to a string, or a tensorlog.matrixdb.MatrixDB: tried to assign %r' % other

  def __iadd__(self,other):
    if self.inner_db is None:
      init_schema = None if self.builder.schema.empty else self.builder.schema
      self.inner_db = matrixdb.MatrixDB(initSchema=init_schema)
      self.inner_db.startBuffers()
    self.inner_db.bufferFile(other) # remember to flush when you use the db!
    return self

class SchemaWrapper(dbschema.TypedSchema):
  """ Subclass a schema to handle a += notation
  """
  def __init__(self,builder):
    super(SchemaWrapper,self).__init__()
    self.empty = False
    self.builder = builder

  def __iadd__(self,other):
    self.empty = False
    if self.builder.db.inner_db is not None:
      logging.warn('Adding to builder.schema after builder.db is initialized, this is dangerous')
    if isinstance(other,parser.Rule):
      for g in other.rhs:
        d = declare.TypeDeclaration(g)
        self.declarePredicateTypes(d.functor, d.args())
    else:
      assert False, 'type declaration error for builder: %s += %s' % (str(self),str(other))
    return self


class RuleWrapper(parser.Rule):
  """ Used by the Builder to hold parts of a rule,
  and combine the parts using operator overloading
  """

  @staticmethod
  def _combine(x,y):
    if x is None: return y
    elif y is None: return x
    else: return x+y
  def __and__(self,other):
    return RuleWrapper(
        None,
        self.rhs + other.rhs,
        features=RuleWrapper._combine(self.features,other.features),
        findall=RuleWrapper._combine(self.findall,other.findall))
  def __or__(self,other):
    return RuleWrapper(None, [], features=self.rhs, findall=other.rhs)
  def __floordiv__(self,other):
    if other.features:
      # self // other.features | other.rhs
      return RuleWrapper(None, self.rhs, features=other.features, findall=other.findall)
    else:
      # self // other.rhs
      return RuleWrapper(None, self.rhs, features=other.rhs)
  def __le__(self,other):
    assert len(self.rhs)==1, 'rule syntax error for builder: %s <= %s' % (str(self),str(other))
    return RuleWrapper(
        self.rhs[0],
        other.rhs,
        features=other.features,
        findall=other.findall)
  def __repr__(self):
    return "RuleWrapper(%r,%r,features=%r,findall=%r" % (self.lhs,self.rhs,self.features,self.findall)

class Options(object):
  """
  For handling options set on the command line.
  """

  def __init__(self):
    pass

  def set_from_command_line(self,argv):
    argspec = ["%s=" % opt_name for opt_name in self.__dict__.keys()]
    optlist,_ = getopt.getopt(argv, 'x', argspec)
    for opt_name,string_val in dict(optlist).items():
      attr_name = opt_name[2:]
      attr_type = type(getattr(self, attr_name))
      if attr_type==type(True):
        # coercing string to bool only is false for empty string, which is unintuitive
        attr_type = lambda s: s not in ['False','false','0','','n','N','No','no']
      setattr(self, attr_name, attr_type(string_val))

  def as_dictionary(self):
    return self.__dict__

  def option_usage(self):
    return " ".join(map(lambda item:"[--%s %r]" % item, self.as_dictionary().items()))

class Experiment(Options):

  def __init__(self):
    self.train_data = 'train.exam'
    self.test_data = 'test.exam'
    self.db = 'db.cfacts'
    self.prog = 'theory.ppr'
    self.mode = 'predict/io'
    self.epochs = 10
    self.batch_size = 125

  def run(self):
    tlog = Compiler(db=self.db, prog=self.prog)
    train = tlog.load_big_dataset(self.train_data)

    loss = tlog.loss(self.mode)
    optimizer = tf.train.AdagradOptimizer(0.1)
    train_step = optimizer.minimize(loss)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    t0 = time.time()
    for i in range(self.epochs):
      b = 0
      for (_,(TX,TY)) in tlog.minibatches(train,batch_size=self.batch_size):
        print 'epoch',i+1,'of',self.epochs,'minibatch',b+1
        train_fd = {tlog.input_placeholder_name(self.mode):TX, tlog.target_output_placeholder_name(self.mode):TY}
        session.run(train_step, feed_dict=train_fd)
        b += 1
        print 'learning time',time.time()-t0,'sec'

    predicted_y = tlog.inference(self.mode)
    actual_y = tlog.target_output_placeholder(self.mode)
    correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    test = tlog.load_small_dataset(self.test_data)
    UX,UY = test[self.mode]
    test_fd = {tlog.input_placeholder_name(self.mode):UX, tlog.target_output_placeholder_name(self.mode):UY}
    acc = session.run(accuracy, feed_dict=test_fd)
    print 'test acc',acc
    return acc

if __name__ == "__main__":
  if len(sys.argv)>=2 and sys.argv[1]=='experiment':
    experiment = Experiment()
    experiment.set_from_command_line(sys.argv[2:])
    experiment.run()
  else:
    print "usage: experiment " + Experiment().option_usage()
