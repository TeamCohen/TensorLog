import sys
import tensorflow as tf

from tensorlog import simple

def runMain(argv):

  # option parsing - options should be passed in as something like sys.argv[1:], eg
  # runMain(["--epochs","20","--stem","cora"])
  opts = simple.Options()
  opts.stem = 'karate'
  opts.regularizer_scale = 0.1
  opts.link_scale = 0.9
  opts.epochs = 20 # 0 for no learning
  opts.max_depth = 4
  opts.learn_friend = True
  opts.learn_label = False
  # override the option defaults, set above
  opts.set_from_command_line(argv)
  # define the input file names from the stems
  factFile = 'inputs/%s.cfacts' % opts.stem
  trainFile = 'inputs/%s-train.exam' % opts.stem
  testFile = 'inputs/%s-test.exam' % opts.stem

  # construct a Compiler object
  tlog = simple.Compiler(db=factFile,prog='social.tlog')

  # tweak the program and database
  tlog.prog.maxDepth = opts.max_depth
  # scale down the friend links, according to the option link_scale.
  # smaller weights are like a higher reset in RWR/PPR
  tlog.db.matEncoding[('friend',2)] = opts.link_scale * tlog.db.matEncoding[('friend',2)]
  # specify which relations will be treated as parameters
  if opts.learn_friend: tlog.mark_db_predicate_trainable('friend/2')
  if opts.learn_label: tlog.mark_db_predicate_trainable('label/2')

  # compile the rules, plus a query mode, into the inference function,
  # which we will use for testing
  mode = 'inferred_label/io'
  predicted_y = tlog.inference(mode)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  # also get the corresponding loss function from tensorlog
  unregularized_loss = tlog.loss(mode)
  # L1 regularize the basic loss function
  weight_vectors = tlog.trainable_db_variables(mode,for_optimization=True)
  regularized_loss = unregularized_loss
  for v in weight_vectors:
    regularized_loss = regularized_loss + opts.regularizer_scale*tf.reduce_sum(tf.abs(v))

  # how to optimize
  optimizer = tf.train.AdagradOptimizer(1.0)
  train_step = optimizer.minimize(regularized_loss)

  # set up the session
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # load the training and test data
  trainData = tlog.load_small_dataset(trainFile)
  testData = tlog.load_small_dataset(testFile)

  # compute initial test-set performance
  (ux,uy) = testData[mode]
  test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
  initial_accuracy = session.run(accuracy, feed_dict=test_fd)
  print 'initial test acc',initial_accuracy

  # run the optimizer for fixed number of epochs
  (tx,ty) = trainData[mode]
  train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
  for i in range(opts.epochs):
    session.run(train_step, feed_dict=train_fd)
    print 'epoch',i+1,'train loss and accuracy',session.run([unregularized_loss,accuracy], feed_dict=train_fd)

  # save the learned model
  tlog.set_all_db_params_to_learned_values(session)
  direc = '/tmp/%s-learned-model.prog' % opts.stem
  tlog.serialize_program(direc)
  print 'learned parameters serialized in',direc

  # compute final test performance
  final_accuracy = session.run(accuracy, feed_dict=test_fd)
  print 'initial test acc',initial_accuracy
  print 'final test acc',final_accuracy

  # return summary of statistics
  return initial_accuracy,final_accuracy

if __name__=="__main__":
  runMain(sys.argv[1:])
