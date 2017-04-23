import logging
import time

import tensorflow as tf
from tensorlog import simple
import expt

def runMain(num=250):
  params = expt.setExptParams(num)
  prog = params['prog']
  tlog = simple.Compiler(db=prog.db, prog=prog, autoset_db_params=False)
  train_data = tlog.load_big_dataset('inputs/train-%d.exam' % num)
  mode = params['targetMode']

  loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(0.1)
  train_step = optimizer.minimize(loss)

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  t0 = time.time()
  epochs = 10
  for i in range(epochs):
      b = 0
      for (_,(TX,TY)) in tlog.minibatches(train_data,batch_size=125):
          print 'epoch',i+1,'of',epochs,'minibatch',b+1
          train_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
          session.run(train_step, feed_dict=train_fd)
          b += 1
  print 'learning time',time.time()-t0,'sec'

  predicted_y = tlog.inference(mode)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  test_data = tlog.load_small_dataset('inputs/test-%d.exam' % num)
  UX,UY = test_data[mode]
  test_fd = {tlog.input_placeholder_name(mode):UX, tlog.target_output_placeholder_name(mode):UY}
  acc = session.run(accuracy, feed_dict=test_fd)
  print 'test acc',acc
  return acc #expect 27.2

if __name__== "__main__":
  runMain()

  

