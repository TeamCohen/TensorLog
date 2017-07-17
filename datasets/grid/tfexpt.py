import time
import tensorflow as tf

from tensorlog import simple
import expt

def setup_tlog(maxD,factFile,trainFile,testFile):
  tlog = simple.Compiler(db=factFile,prog="grid.ppr")
  tlog.prog.db.markAsParameter('edge',2)
  tlog.prog.maxDepth = maxD
  trainData = tlog.load_small_dataset(trainFile)
  testData = tlog.load_small_dataset(testFile)
  return (tlog,trainData,testData)

def trainAndTest(tlog,trainData,testData,epochs):
  mode = 'path/io'
  predicted_y = tlog.inference(mode)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  unregularized_loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(1.0)
  train_step = optimizer.minimize(unregularized_loss)

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  (tx,ty) = trainData[mode]
  train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
  t0 = time.time()
  for i in range(epochs):
    print 'epoch',i+1
    session.run(train_step, feed_dict=train_fd)
  print 'learning takes',time.time()-t0,'sec'
  (ux,uy) = testData[mode]
  test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
  acc = session.run(accuracy, feed_dict=test_fd)
  print 'test acc',acc
  return acc


def runMain():
  (goal,n,maxD,epochs) = expt.getargs()
  assert goal=="acc"
  (factFile,trainFile,testFile) = expt.genInputs(n)
  (tlog,trainData,testData) = setup_tlog(maxD,factFile,trainFile,testFile)
  trainAndTest(tlog,trainData,testData,epochs)


if __name__=="__main__":
  runMain()
