import tensorflow as tf

from tensorlog import simple
import expt

def runMain():
  # generate the data for a 10-by-10 grid
  (factFile,trainFile,testFile) = expt.genInputs(16)

  # generate the rules - for transitive closure
  b = simple.Builder()
  path,edge = b.predicates("path,edge")
  X,Y,Z = b.variables("X,Y,Z")
  b.rules += path(X,Y) <= edge(X,Y)
  b.rules += path(X,Y) <= edge(X,Z) & path(Z,Y)

  # construct a Compiler object
  tlog = simple.Compiler(db=factFile,prog=b.rules)

  # configure the database so that edge weights are a parameter
  tlog.prog.db.markAsParameter('edge',2)
  # configure the program so that maximum recursive depth is 16
  tlog.prog.maxDepth = 16

  # compile the rules, plus a query mode, into the inference function
  # we want to optimize - queries of the form {Y:path(x,Y)} where x is
  # a given starting point in the grid (an input) and Y is an output
  mode = 'path/io'
  predicted_y = tlog.inference(mode)

  # when we ask for an inference function, Tensorlog also compiles a
  # loss function.  ask for the placeholder used to hold the desired
  # output when we're computing loss, and use that to define an
  # accuracy metric, for testing
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  # now get up the loss used in learning from the compiler and set up
  # a learner for it
  unregularized_loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(1.0)
  train_step = optimizer.minimize(unregularized_loss)

  # set up the session
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # load the training and test data
  trainData = tlog.load_small_dataset(trainFile)
  testData = tlog.load_small_dataset(testFile)

  # run the optimizer for 20 epochs
  (tx,ty) = trainData[mode]
  train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
  for i in range(20):
    session.run(train_step, feed_dict=train_fd)
    print('epoch',i+1,'train loss and accuracy',session.run([unregularized_loss,accuracy], feed_dict=train_fd))

  # test performance
  (ux,uy) = testData[mode]
  test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
  acc = session.run(accuracy, feed_dict=test_fd)

  print('test acc',acc)
  return acc

if __name__=="__main__":
  runMain()
