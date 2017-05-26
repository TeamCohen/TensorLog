import time
import sys
import getopt
import os
import logging

# Demo code illustrating how to run a learn experiment with Tensorlog and Tensorflow
#
# Invoke this script with these sorts of command lines (for other options see the code)
#
#   python demo.py --action train --epochs 50 --num 10000 --batch_size 2500
#   python demo.py --action test --num 1000
#
# In my experiments this particular sequence of command gets you an error of about 4.9%
#
#   python demo.py --action train --epochs 50 --num 10000
#   python demo.py --action test --num 10000
#

import tensorflow as tf
from tensorlog import simple

RELS = "directed_by has_genre has_imdb_rating has_imdb_votes has_tags release_year starred_actors written_by in_language".split(" ")

def configure_from_command_line(argv):
  """Create an object c which is holds all the user-defined options for
  this experiment.
  """
  class ExptOpts(object):
    pass
  c = ExptOpts()
  c.action = 'train' # could also be 'test'

  # options used for training
  c.batch_size = 100 # actually minibatch size
  c.rate = 0.1 # passed to AdagradOptimizer
  c.epochs = 5
  c.num = 1000 # used to pick the training/test set files and database files

  # options used for testing
  c.model = 'learned-model.db'

  # override the default options for 'c' using command line arguments,
  # using Python's reflection capabilities
  argspec = ["%s=" % opt_name for opt_name in c.__dict__.keys()]
  optlist,_ = getopt.getopt(argv, 'x', argspec)
  for opt_name,string_val in dict(optlist).items():
    attr_name = opt_name[2:]
    attr_type = type(getattr(c,attr_name))
    setattr(c, attr_name, attr_type(string_val))
  # echo the current options to stdout
  for attr_name,attr_val in sorted(c.__dict__.items()):
    print 'option:',attr_name,'set to',attr_val
  return c

def generate_rules():
  """Generate rules from the list of known relations, which will be
  returned in a simple.Builder object.
  """
  b = simple.Builder()
  answer,mentions_entity,has_feature = b.predicates("answer mentions_entity has_feature")
  Question,Movie,Entity,F = b.variables("Question Movie Entity F")
  # rules for answering questions like 'what movie was directed by FOO?'
  for rel in RELS:
    P_rel = b.predicate(rel)
    W_rel = b.predicate('wBack_%s' % rel)
    b += answer(Question,Movie) <= mentions_entity(Question,Entity) & P_rel(Movie,Entity) // (W_rel(F) | has_feature(Question,F))
  # for questions like 'who directed FOO?'
  for rel in RELS:
    P_rel = b.predicate(rel)
    W_rel = b.predicate('wFore_%s' % rel)
    b += answer(Question,Entity) <= mentions_entity(Question,Movie) & P_rel(Movie,Entity) // (W_rel(F) | has_feature(Question,F))
  # echo rules to stdout
  b.rules.listing()
  return b

def run_main():
  logging.basicConfig(level=logging.DEBUG)

  t0 = time.time()

  # configure the experiment, generate the rules, and initialize the
  # Tensorlog compiler.  If we're training, then also initialize the
  # weight vectors to some sort of default values.
  c = configure_from_command_line(sys.argv[1:])
  b = generate_rules()
  # databases can be stored in two formats: the .db format or the
  # .cfacts format.  A .cfacts file is basically a tab-separated-value
  # file, where the first column is a relation name, the other columns
  # are arguments to that relation, and the final column is a weight
  # (if it's a number). The .cfacts file can also include typing
  # information, in lines like '# :-
  # mentions_entity(question_t,entity_t)' .cfacts files must be sorted
  # by relation type.  The .db format is a binary format which is more
  # compact and faster to load.  The syntax "foo.db|foo.cfacts" for a
  # database tells Tensorlog to load a cached .db version of the
  # .cfacts file if it exists (and is more recent than the .cfacts
  # file) and otherwise to load the .cfacts file and create a cached
  # version in the .db file.
  dbspec = '/tmp/train-%d.db|inputs/train-%d.cfacts' % (c.num,c.num) if c.action=='train' else c.model
  tlog = simple.Compiler(db=dbspec, prog=b.rules, autoset_db_params=(c.action=='train'))

  # set up the optimizer
  mode = 'answer/io'
  unregularized_loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(c.rate)
  train_step = optimizer.minimize(unregularized_loss)

  # define the measure we'll use to report quality of a learned model
  predicted_y = tlog.inference(mode)  # inference is the
                                      # proof-counting semantics
                                      # followed by a softmax
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  # initialize the tensorflow session
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  t1 = time.time()
  print 'compilation and session initialization',(t1-t0)/1000.0,'sec'

  if c.action=='test':

    # a small_dataset is just a dictionary mapping function names (ie
    # modes), like "answer/io", to pairs X,Y, where X is an input and
    # Y a desired output.  If the action is to 'test' a learned model
    # then load in the test data and find that x,y pair.
    test_data = tlog.load_small_dataset('inputs/test-%d.exam' % c.num)
    _,(x,y) = test_data.items()[0]
    # ... then compute error rate and print it
    test_batch_fd = {tlog.input_placeholder_name(mode):x,
                     tlog.target_output_placeholder(mode):y}
    print 'test error',100*(1.0 - session.run(accuracy, feed_dict=test_batch_fd)),'%'

  else:
    assert c.action=='train'

    # load_big_dataset returns an object which can enumerate
    # mini-batches.  The object holds all the input and output vectors
    # for tensorlog as sparse vectors (ie, it doesn't stream thru them
    # from disk). This is still important from memory usage point of
    # view, however, because we cannot encode these x,y pairs as
    # sparse in tensorflow, since tensorflow doesn't support sparse
    # matrix-sparse matrix product, only dense matrix-sparse matrix
    # product.  So a the big dataset object will convert each
    # minibatch to a dense format on-the-fly before training on it.
    train_data = tlog.load_big_dataset('inputs/train-%d.exam' % c.num)

    t2 = time.time()
    print 'data loading',(t2-t1),'sec'

    # finally, run the learner for a fixed number of epochs
    for i in range(c.epochs):
      print 'starting epoch',i+1,'of',c.epochs,'...'
      b = 0
      for _,(x,y) in tlog.minibatches(train_data,batch_size=c.batch_size):
        train_batch_fd = {tlog.input_placeholder_name(mode):x, tlog.target_output_placeholder_name(mode):y}
        session.run(train_step, feed_dict=train_batch_fd)
        print 'finished minibatch',b+1,'epoch',i+1,'cumulative training time',(time.time()-t2),'sec'
        b += 1
    t3 = time.time()
    print 'learning',(t3-t2),'sec'

    # We have now learned values for all the parameters. This command
    # copies those learned values back into the knowledge
    # graph/database maintained by Tensorlog.
    tlog.set_all_db_params_to_learned_values(session)

    # Finally, write the whole knowledge graph, including the learned
    # parameters, out to disk in a compact format, which can be read
    # back in when we use the 'test' action
    tlog.serialize_db('learned-model.db')
    print 'wrote learned model to learned-model.db'

if __name__ == "__main__":
  run_main()
