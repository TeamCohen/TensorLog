import logging
import time

import tensorflow as tf

from tensorlog import simple

import expt

def runMain(saveInPropprFormat=True):
  params = expt.setExptParams()
  prog = params['prog']
  tlog = simple.Compiler(db=prog.db, prog=prog, autoset_db_params=False)
  train_data = tlog.load_small_dataset('inputs/train.examples')
  test_data = tlog.load_small_dataset('inputs/test.examples')

  mode = 'samebib/io'
  TX,TY = train_data[mode]
  UX,UY = test_data[mode]
  loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(0.1)
  train_step = optimizer.minimize(loss)
  train_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
  test_fd = {tlog.input_placeholder_name(mode):UX, tlog.target_output_placeholder_name(mode):UY}

  t0 = time.time()
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  epochs = 30
  for i in range(epochs):
    # progress
    print 'epoch',i+1,'of',epochs
    session.run(train_step, feed_dict=train_fd)
  print 'learning time',time.time()-t0,'sec'

  inference = tlog.inference(mode)
  predicted_y = session.run(inference, feed_dict=test_fd)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  if saveInPropprFormat:
    # save test results in ProPPR format
    from tensorlog import declare
    from tensorlog import dataset
    from tensorlog import expt as tlog_expt
    m = declare.asMode(mode)
    native_test_data = dataset.Dataset({m:tlog.xc._unwrapOutput(UX)},{m:tlog.xc._unwrapOutput(UY)})
    savedTestExamples = 'tmp-cache/cora-test.examples'
    savedTestPredictions = 'tmp-cache/cora-test.solutions.txt'
    native_test_data.saveProPPRExamples(savedTestExamples,tlog.db)
    tlog_expt.Expt.predictionAsProPPRSolutions(
        savedTestPredictions,'samebib',tlog.db,tlog.xc._unwrapOutput(UX),tlog.xc._unwrapOutput(predicted_y))
    print 'ready for commands like: proppr eval %s %s --metric auc --defaultNeg' % (savedTestExamples,savedTestPredictions)

  acc = session.run(accuracy, feed_dict=test_fd)
  print 'test acc',acc
  return acc

if __name__=="__main__":
  runMain()
