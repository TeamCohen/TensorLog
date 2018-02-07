import os
import time
import tensorflow as tf
import numpy as np

from tensorlog import simple
from tensorlog import matrixdb
from tensorlog import diffruleplugin as diffrule


def runMain():
  matrixdb.conf.ignore_types = True
  dbfile = "inputs/train-1000.cfacts"
  progfile = "diffrule-theory.ppr"
  
  tlog = simple.Compiler(
      db=dbfile,
      prog=progfile)

  trainData = tlog.load_big_dataset("inputs/train-1000.exam")
  testData  = tlog.load_big_dataset("inputs/test-1000.exam")

  
  drmode = trainData.modesToLearn()[0]
  option = diffrule.defaultOptions()
  
  dr = diffrule.insertDRIntoXC(tlog.xc, drmode, option)
  loss = tlog.loss(drmode)
  optimizer = tf.train.AdagradOptimizer(0.1)
  train_step = optimizer.minimize(loss)

  
  saver = tf.train.Saver(max_to_keep=option.max_epoch)
  saver = tf.train.Saver()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = False
  config.log_device_placement = False
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
      tf.set_random_seed(option.seed)
      sess.run(tf.global_variables_initializer())
      print("Session initialized.")
      
      predicted_y = tlog.inference(drmode)
      actual_y = tlog.target_output_placeholder(drmode)
      correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
      testData.shuffle()
      UX,UY=tlog.xc.wrapInput(testData.getX(drmode)),tlog.xc.wrapInput(testData.getY(drmode))
      def test(X,Y,name='test'):
        test_fd = {tlog.input_placeholder_name(drmode):X,
                   dr.queries: np.zeros(dtype=np.int32,shape=[X.shape[0],dr.num_step]), 
                   tlog.target_output_placeholder_name(drmode):Y}
        acc,in_top = sess.run([accuracy,dr.in_top], feed_dict=test_fd)
        print "%s acc %f in_top %f" % (name,acc,np.mean(in_top))

      test(UX,UY) 
      t0 = time.time()
      for i in range(option.max_epoch):
        print 'epoch',i+1,'of',option.max_epoch
        b = 0
        for (_,(TX,TY)) in tlog.minibatches(trainData):
          train_fd = {tlog.input_placeholder_name(drmode):TX, 
                      dr.queries: np.zeros(dtype=np.int32,shape=[TX.shape[0],dr.num_step]), 
                      tlog.target_output_placeholder_name(drmode):TY}
          sess.run(train_step, feed_dict=train_fd)
          b += 1
        #test(VX,VY,'validation')
        print 'learning time',time.time()-t0,'sec'
      
      test(UX,UY)
  print("="*36 + "Finish" + "="*36)   

  
if __name__=='__main__':
  runMain()
  