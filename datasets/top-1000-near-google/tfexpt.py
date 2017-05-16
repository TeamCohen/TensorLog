import logging
import time

import tensorflow as tf
from tensorlog import simple
import expt

def runMain():
  params = expt.setExptParams()
  prog = params['prog']
  tlog = simple.Compiler(db=prog.db, prog=prog, autoset_db_params=False)
  train_data = tlog.annotate_big_dataset(params['trainData'])
  test_data = tlog.annotate_small_dataset(params['testData'])
  result={}
  for mode in params['trainData'].modesToLearn():
    print mode
    loss = tlog.loss(mode)
    optimizer = tf.train.AdagradOptimizer(0.1)
    train_step = optimizer.minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    t0 = time.time()
    epochs = 10
    for i in range(epochs):
        b = 0
        print 'epoch',i+1,'of',epochs
        for (_,(TX,TY)) in tlog.minibatches(train_data,batch_size=125):
            #print 'epoch',i+1,'of',epochs,'minibatch',b+1
            train_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}
            session.run(train_step, feed_dict=train_fd)
            b += 1
    print 'learning time',time.time()-t0,'sec'

    predicted_y = tlog.inference(mode)
    actual_y = tlog.target_output_placeholder(mode)
    correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    if mode in test_data:
      UX,UY = test_data[mode]
      test_fd = {tlog.input_placeholder_name(mode):UX, tlog.target_output_placeholder_name(mode):UY}
      acc = session.run(accuracy, feed_dict=test_fd)
      print mode,'test acc',acc
      result[mode]= acc
  return result

if __name__== "__main__":
  runMain()
