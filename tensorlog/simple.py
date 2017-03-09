import logging
import os.path

from tensorlog import comline
from tensorlog import declare
from tensorlog import dataset
from tensorlog import funs
from tensorlog import matrixdb
from tensorlog import program
from tensorlog import tensorflowxcomp
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
        automatically. This usually works for rule_features but but
        others.

      summary_file: if not None, and if target=='tensorflow', this
        location will be used as to hold summary data for tensorboard
        on the tensorlog operations.
    """

    # parse the db argument
    if isinstance(db,matrixdb.MatrixDB):
      pass
    elif isinstance(db,str):
      self.db = comline.parseDBSpec(db)
    else:
      assert False,'cannot convert %r to a database' % db

    # parse the program argument
    if isinstance(prog,program.Program):
      pass
    elif isinstance(prog,str):
      self.prog = comline.parseProgSpec(prog,self.db,proppr=rule_features)
      if autoset_db_params:
        self.prog.setAllWeights()
    else:
      assert False,'cannot convert %r to a program' % prog

    # parse the target argument
    self.target = target
    if target=='tensorflow':
      self.xc = tensorflowxcomp.SparseMatDenseMsgCrossCompiler(self.prog, summaryFile=summary_file)
    elif target=='theano':
      self.xc = theanoxcomp.SparseMatDenseMsgCrossCompiler(self.prog)
    else:
      assert False,'illegal target %r: valid targets are "tensorflow" and "theano"' % target

  def inference(self,mode):
    """ An expression for the inference associated with a mode
    """
    args,expr = self.xc.inference(declare.asMode(mode))
    return expr

  def loss(self,mode):
    """ An expression for the unregularized loss associated with a mode
    """
    args,expr = self.xc.dataLoss(declare.asMode(mode))
    return expr

  def trainable_db_variables(self,mode):
    """ The trainable variables associated with the tensorlog database
    """
    return self.xc.getParamVariables(declare.asMode(mode))

  #
  # needed for building feed_dicts for training/testing tensorflow
  # TOFIX - can I get these from the arguments to loss, inference?
  #

  def input_placeholder_name(self,mode):
    """ For tensorflow, the placeholder associated with the input to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getInputName(declare.asMode(mode))

  def target_output_placeholder_name(self,mode):
    """ For tensorflow, the placeholder associated with the output to this function.
    """
    assert self.target == 'tensorflow'
    return self.xc.getTargetOutputName(declare.asMode(mode))

  #
  # needed if you don't want to autoset the parameters stored in tensorlog's db
  #

  def param_list(self):
    """ Identifiers for trainable tensorlog DB relations. """
    return self.prog.getParamList()

  def param_is_set(self,param_id):
    """ Test to see if a parameter relation has a value. """
    (functor,arity) = param_id
    return self.db.parameterIsInitialized(functor,arity)

  def get_param_value(self,param_id):
    """ Get the value of a parameter relation has a value. """
    (functor,arity) = param_id
    return self.db.getParameter(functor,arity)

  def set_param_value(self,param_id,value):
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
  # expose other useful routines
  #

  def _mode_as_string(self,mode): return mode.getFunctor() + "/" + "".join(mode.arg(i) for i in range(mode.getArity()))

  def load_dataset(self,dataset_spec):
    """Return a dictionary where keys are strings defining tensorlog
    functions - e.g., answer/io - and the values are pairs (X,Y) where
    X is a matrix that can be used as a batch input to the inference
    function, and Y is a matrix that is the desired output.

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
    dset = comline.parseDatasetSpec(dataset_spec,self.db)
    # convert to something bereft of tensorlog data structures: a
    # dictionary mapping strings like "p/io" to X,Y pairs, where X and
    # Y are wrapped inputs.
    def wrapped_xy_pair(mode): return (self.xc.wrapInput(dset.getX(mode)), self.xc.wrapInput(dset.getY(mode)))
    return dict((self._mode_as_string(mode),wrapped_xy_pair(mode)) for mode in dset.modesToLearn())

  def minibatches(self,dataset_dict,batch_size=100,shuffle_first=True):
    """Yields a series of pairs (mode,(X,Y)) where X and Y are a
    minibatch suitable for training the function designated by mode.
    """
    x_dict = {}
    y_dict = {}
    for mode_str,(x,y) in dataset_dict.items():
      mode = declare.asMode(mode_str)
      x_dict[mode] = self.xc.unwrapInput(x)
      y_dict[mode] = self.xc.unwrapInput(y)
      dset = dataset.Dataset(x_dict,y_dict)
    for mode,bx,by in dset.minibatchIterator(batchSize=batch_size,shuffleFirst=shuffle_first):
      yield self._mode_as_string(mode),(self.xc.wrapInput(bx),self.xc.wrapInput(by))
