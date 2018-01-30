import tensorflow as tf
import numpy as np
import argparse
import os

from tensorlog import xcomp
from tensorlog import tensorflowxcomp as tfx

_TFZERO=tf.constant(0,dtype=tf.float32)

class DiffRule(object):
  def __init__(self, xc, mode, option):
    self.xc = xc
    self.mode = mode
    self.num_step = option.num_step
    self.num_query = xc.db.numMatrices()
    self.num_entity = xc.db.dim()
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
    
  def _random_uniform_unit(self, r, c):
    """ Initialize random and unit row norm matrix of size (r, c). """
    bound = 6./ np.sqrt(c)
    init_matrix = np.random.uniform(-bound, bound, (r, c))
    init_matrix = np.array(map(lambda row: row / np.linalg.norm(row), init_matrix))
    return init_matrix
  def _onehot_to_indices(self,onehots):
    return tf.where(tf.not_equal(onehots, _TFZERO))[:,1]
  def _build_input(self):
    self.sources = self.xc._createPlaceholder('diffrule_X','vector',None)
    self.tails = self._onehot_to_indices(self.sources)

    self.targets = self.xc._createPlaceholder(xcomp.TRAINING_TARGET_VARNAME,'vector',None)
    self.heads = self._onehot_to_indices(self.targets)
    
    # typeless for now; we just need the max size
    self.num_operator = len(self.xc.possibleOps(self.sources))
        
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
    
    # memories: (will be) a tensor of size (batch_size, t+1, num_entity),
    # where t is the current step (zero indexed)
    # Then tensor represents currently populated memory cells.
    memories = tf.expand_dims(
                     tf.one_hot(
                            indices=self.tails,
                            depth=self.num_entity), 1) 
    
    for t in xrange(self.num_step):
      attention_memories.append(
                      tf.nn.softmax(
                      tf.squeeze(
                          tf.matmul(
                              tf.expand_dims(rnn_outputs[t], 1),
                              tf.stack(rnn_outputs[0:t + 1], axis=2)),
                      squeeze_dims=[1])))
      
      # memory_read: tensor of size (batch_size, num_entity)
      memory_read = tf.squeeze(
                      tf.matmul(
                          tf.expand_dims(attention_memories[t], 1),
                          memories),
                      squeeze_dims=[1])
      
      if t < self.num_step - 1:
        # database_results: (will be) a list of num_operator tensors,
        # each of size (batch_size, num_entity).
        database_results = []
        # possibleOps returns a list of expressions that have each been
        # multiplied by memory_read already
        operators = self.xc.possibleOps(memory_read)
        for r,op_expr in enumerate(operators):
          op_attn = attention_operators[t][r]
          dri = op_expr * op_attn
          database_results.append(dri)

        added_database_results = tf.add_n(database_results)
        if self.norm:
            added_database_results /= tf.maximum(self.thr, tf.reduce_sum(added_database_results, axis=1, keep_dims=True))                
        
        if self.dropout > 0.:
          added_database_results = tf.nn.dropout(added_database_results, keep_prob=1. - self.dropout)

        # Populate a new cell in memory by concatenating.  
        memories = tf.concat(
            [memories,
            tf.expand_dims(added_database_results, 1)],
            axis=1)
      else:
        # predictions: tensor of size (batch_size, num_entity)
        self.predictions = memory_read
                       
    self.final_loss = -tf.reduce_sum(self.targets * tf.log(tf.maximum(self.predictions, self.thr)), 1)
     
    if not self.accuracy:
      self.in_top = tf.nn.in_top_k(
                      predictions=self.predictions,
                      targets=self.heads,
                      k=self.top_k)
    else: 
      _, indices = tf.nn.top_k(self.predictions, self.top_k, sorted=False)
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
    
