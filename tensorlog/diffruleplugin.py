import tensorflow as tf
import numpy as np
import argparse
import os
import time
import logging

from tensorlog import xcomp
from tensorlog import tensorflowxcomp as tfx
from tensorlog import symtab
from tensorlog import config

conf = config.Config()

_TFZERO=tf.constant(0,dtype=tf.float32)
_TFEYE2=tf.eye(2,dtype=tf.float32)

class DiffRule(object):
  def __init__(self, xc, mode, option):
    self.xc = xc
    self.mode = mode
    self.num_step = option.num_step
    self.num_query = xc.db.numMatrices()
    self.query_embed_size = option.query_embed_size
    self.rnn_state_size = option.rnn_state_size
    self.num_layer = option.num_layer
    self.top_k = option.top_k
    self.thr = option.thr
    self.dropout = option.dropout
    self.learning_rate = option.learning_rate
    self.accuracy = option.accuracy
    self.seed = option.seed
    self.norm = not option.no_norm
    self.typetab = symtab.SymbolTable(xc.db.schema.getTypes())
    self._domain = xc.db.schema.getDomain(mode.getFunctor(),mode.getArity())
    self._range = xc.db.schema.getRange(mode.getFunctor(),mode.getArity())
    logging.debug("DiffRule domain: %s"%self._domain)
    logging.debug(" DiffRule range: %s"%self._range)
    
  def _random_uniform_unit(self, r, c):
    """ Initialize random and unit row norm matrix of size (r, c). """
    bound = 6./ np.sqrt(c)
    init_matrix = np.random.uniform(-bound, bound, (r, c))
    init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
    return init_matrix
  def _nhot_to_indices(self,nhots):
    return tf.where(tf.not_equal(nhots, _TFZERO))
  def _build_input(self):
    self.sources = self.xc._createPlaceholder('diffrule_X','vector',self._domain)
    self.tails = self._nhot_to_indices(self.sources)[:,1]

    self.targets = self.xc._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',self._range)
    target_indices = self._nhot_to_indices(self.targets)
    self.heads = target_indices[:,1]
    # multiclass_adapter: matrix s.t. MP duplicates the rows of
    # the prediction matrix wherever a query has >1 solution
    self.multiclass_adapter = tf.one_hot(indices=target_indices[:,0],depth=tf.shape(self.targets)[0])
    
    # use a dummy subExpr here; we just need the total size
    self.num_operator = sum(len(self.xc.possibleOps(_TFEYE2,_type)) for _type in self.xc.db.schema.getTypes())
    
    # NB: we probably don't need this subsystem anymore..?
    self.queries = tf.placeholder(tf.int32, [None, self.num_step], 'diffrule_query')
    query_embedding_params = tf.Variable(self._random_uniform_unit(
                                                  self.num_query + 1,  # <END> token 
                                                  self.query_embed_size),
                                              dtype=tf.float32,name="diff-rule_query-embedding")

    rnn_inputs = tf.nn.embedding_lookup(query_embedding_params,
                                        self.queries)
    return rnn_inputs
  def _build_graph(self):
    """ Build a computation graph that represents the model """
    rnn_inputs = self._build_input()                        
    # rnn_inputs: a list of num_step tensors,
    # each tensor of size (batch_size, query_embed_size).    
    rnn_inputs = [tf.reshape(q, [-1, self.query_embed_size]) 
                       for q in tf.split(rnn_inputs,
                                         self.num_step,
                                         axis=1)]
    
    cell = tf.contrib.rnn.LSTMCell(self.rnn_state_size,
                                                 state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell(
                                                [cell] * self.num_layer,
                                                state_is_tuple=True)
    init_state = cell.zero_state(tf.shape(self.tails)[0], tf.float32)
    
    # rnn_outputs: a list of num_step tensors,
    # each tensor of size (batch_size, rnn_state_size).
    rnn_outputs, final_state = tf.contrib.rnn.static_rnn(
                                            cell,
                                            rnn_inputs,
                                            initial_state=init_state)
    
    W = tf.Variable(np.random.randn(
                            self.rnn_state_size,
                            self.num_operator),
                        dtype=tf.float32,name="diff-rule_W")
    b = tf.Variable(np.zeros(
                            (1, self.num_operator)),
                        dtype=tf.float32,name="diff-rule_b")

    # attention_operators: a list of num_step lists,
    # each inner list has num_operator tensors,
    # each tensor of size (batch_size, 1).
    # Each tensor represents the attention over an operator. 
    attention_operators = [tf.split(
                                tf.nn.softmax(
                                  tf.matmul(rnn_output, W) + b),
                                self.num_operator,
                                axis=1) 
                                for rnn_output in rnn_outputs]
    
    # attention_memories: (will be) a list of num_step tensors,
    # each of size (batch_size, t+1),
    # where t is the current step (zero indexed).
    # Each tensor represents the attention over currently populated memory cells. 
    attention_memories = []
    
    # memories: (will be) for each type,
    # a tensor of size (batch_size, t+1, num_entity),
    # where t is the current step (zero indexed)
    # The tensor represents currently populated memory cells.
    memories = {}
    # fill all types with zero first
    for _type in self.xc.db.schema.getTypes():
      #                           batch_size             t+1  num_entity
      memories[_type] = tf.zeros([tf.shape(self.tails)[0], 1, self.xc.db.dim(_type)])
    # then initialize the domain memory with the input distribution
    memories[self._domain]= tf.expand_dims(
                     tf.one_hot(
                            indices=self.tails,
                            depth=self.xc.db.dim(self._domain)), 1) 
    
    def read_from_memory(read_t,read_type):
      """Get the memory matrix at a time t for a particular db type"""
      return tf.squeeze(
        tf.matmul(
          tf.expand_dims(attention_memories[read_t], 1),
          memories[read_type]),
        squeeze_dims=[1]) 

    for t in xrange(self.num_step):
      attention_memories.append(
                      tf.nn.softmax(
                      tf.squeeze(
                          tf.matmul(
                              tf.expand_dims(rnn_outputs[t], 1),
                              tf.stack(rnn_outputs[0:t + 1], axis=2)),
                      squeeze_dims=[1])))
      
      logging.debug( "t= %d of %d"%(t,self.num_step-1))
      logging.debug( " attention_memories %d %s"%(len(attention_memories),str(attention_memories[-1].shape)))
      logging.debug( " memories\n *%s"%"\n *".join(str(m.shape) for m in memories.values()))
      
      offset=0 # keep track of our place in attention_operators
      
      if t < self.num_step - 1:
        logging.debug( " chaining through the database")

        # database_results: (will be) 
        # for each output type,
        # a list of num_operator tensors,
        # each of size (batch_size, num_entity).
        #
        # we need to accumulate weight on output types 
        # from all available input types, so this has to persist across 
        # the input type loop.
        database_results = {}
        for input_type in memories.keys():
          logging.debug( "  from %s"% input_type)
          # memory_read: a tensor of size (batch_size, num_entity[input_type])
          memory_read = read_from_memory(t,input_type)

          # possibleOps returns a list of expressions that have each been
          # multiplied by memory_read already
          operators = self.xc.possibleOps(memory_read, input_type)
          for r,op_possibility in enumerate(operators):
            # offset ensures operators don't overlap between input types
            op_attn = attention_operators[t][offset+r]
            op_expr,op_outputType = (op_possibility,self._range) if self.xc.db.isTypeless() else op_possibility
            if op_outputType not in database_results:
              database_results[op_outputType] = []
            dri = op_expr * op_attn
            database_results[op_outputType].append(dri)
          offset += len(operators)
        # endfor input type 

        # now sum over all results for each output type
        # regardless of input type
        added_database_results = {}
        for _ot in database_results:
          added_database_results[_ot]=tf.add_n(database_results[_ot])
          
          if self.norm:
            added_database_results[_ot] /= tf.maximum(self.thr, tf.reduce_sum(added_database_results[_ot], axis=1, keep_dims=True))                

          if self.dropout > 0.:
            added_database_results[_ot] = tf.nn.dropout(added_database_results[_ot], keep_prob=1. - self.dropout)

        # Populate a new cell in memory by concatenating.  
        for _ot in added_database_results:
          logging.debug( "  to %s"%_ot)
          memories[_ot] = tf.concat(
            [memories[_ot],
             tf.expand_dims(added_database_results[_ot], 1)],
            axis=1)
      else: # t == num_step-1
        logging.debug( " saving predictions")
        # predictions: tensor of size (batch_size, num_entity[_range])
        self.predictions = read_from_memory(t,self._range)
        # mc_predictions: tensor of size (|self.heads|, num_entity[_range])
        self.mc_predictions = tf.matmul(self.multiclass_adapter,self.predictions)
    
    self.final_loss = -tf.reduce_sum(self.targets * tf.log(tf.maximum(self.predictions, self.thr)), 1)
     
    if not self.accuracy:
      # (this is the default computation)
      self.in_top = tf.nn.in_top_k(
                      predictions=self.mc_predictions,
                      targets=self.heads,
                      k=self.top_k)
    else: 
      _, indices = tf.nn.top_k(self.mc_predictions, self.top_k, sorted=False)
      # NB: this will underestimate accuracy in proportion to the average number of solutions per query
      self.in_top = tf.equal(tf.squeeze(indices), self.heads) 
     
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    gvs = self.optimizer.compute_gradients(tf.reduce_mean(self.final_loss))
    # for debugging, a message to print the gradient variables
    # print "\n".join("%s" % (str(v)) for (g,v) in gvs if g != None)
    capped_gvs = map(
      lambda (grad, var): self._clip_if_not_None(grad, var, -5., 5.), 
      gvs) 
    self.optimizer_step = self.optimizer.apply_gradients(capped_gvs)
  def _clip_if_not_None(self, g, v, low, high):
    """ Clip not-None gradients to (low, high). """
    """ Gradient of T is None if T not connected to the objective. """
    if g is not None:
        return (tf.clip_by_value(g, low, high), v)
    else:
        return (g, v)
      
class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))
def make_parser():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--no_rules', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)    
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    # data property
    parser.add_argument('--datadir', default="", type=str)
    parser.add_argument('--resplit', default=False, action="store_true")
    parser.add_argument('--no_link_percent', default=0., type=float)
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    # model architecture
    parser.add_argument('--num_step', default=3, type=int)
    parser.add_argument('--num_layer', default=1, type=int)
    parser.add_argument('--rnn_state_size', default=128, type=int)
    parser.add_argument('--query_embed_size', default=128, type=int)
    # optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--min_epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)
    parser.add_argument('--dropout', default=0., type=float)
    # evaluation
    parser.add_argument('--get_phead', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    return parser

def defaultOptions():
  parser = make_parser()
  d = vars(parser.parse_args())
  option = Option(d)
#   print "Config"
#   print "\n".join("%s: %s" % (str(k),str(v)) for k,v in d.iteritems())
  
  option.tag = time.strftime("%y-%m-%d-%H-%M")
  option.datadir="diffruleplugin"
  option.this_expsdir = os.path.join(os.path.join(option.datadir, "exps"), option.tag)
  if not os.path.exists(option.this_expsdir):
      os.makedirs(option.this_expsdir)
  option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
  if not os.path.exists(option.ckpt_dir):
      os.makedirs(option.ckpt_dir)
  option.model_path = os.path.join(option.ckpt_dir, "model")
  option.save()
  return option

def insertDRIntoXC(xc,drmode,options):
    """
    insertDRIntoXC: Adds diff-rule into an existing tensorflow cross-compiler at the specified mode.
    returns: a DiffRule model object (which you will need for amending feeds; see kin-ablation example)
    """
    assert isinstance(xc,tfx.TensorFlowCrossCompiler), "I only know how to mix diff-rule with tensorflow cross-compilers"
    # pulled from ensureCompiled()
    xc._wsDict[drmode] = xc.ws = xcomp.Workspace(xc)
    # pulled from _doCompile
    xc._setupGlobals()
    
    # assemble inference function:
    dr = DiffRule(xc,drmode,options)
    dr._build_graph()
    xc._wsDict[drmode].inferenceExpr = dr.predictions
    xc._wsDict[drmode].inferenceArgs = [dr.sources] # TODO: may need to fix queries
    xc._wsDict[drmode].inferenceOutputType = None # same as _onlyType for now
    
    # assemble loss function:
    xc._wsDict[drmode].dataLossArgs = xc._wsDict[drmode].inferenceArgs + [dr.targets]
    xc._wsDict[drmode].dataLossExpr = dr.final_loss
    
    # pulled from _finalizeCompile
    if xc.summaryFile:
      xc.summaryMergeAll = tf.summary.merge_all()
    return dr    
    
