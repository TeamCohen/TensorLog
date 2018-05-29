# todo: use minibatch size

BATCH_SIZE=250

import time
import tensorflow as tf

from tensorlog import simple
import bigexpt

def setup_tlog(maxD,factFile,trainFile,testFile):
  tlog = simple.Compiler(db=factFile,prog="grid.ppr")
  tlog.prog.db.markAsParameter('edge',2)
  tlog.prog.maxDepth = maxD
  trainData = tlog.load_small_dataset(trainFile)
  testData = tlog.load_small_dataset(testFile)
  return (tlog,trainData,testData)

# run timing experiment
def timingExpt(tlog,maxD,trainFile,minibatch):
    print 'depth',maxD,'minibatch',minibatch
    tlog.prog.maxDepth = maxD
    dset = tlog.load_dataset(trainFile)
    predicted_y = tlog.inference('path/io')
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    t0 = time.time()
    for mode,(tx,ty) in tlog.minibatches(dset,batch_size=minibatch):
      train_fd = {tlog.input_placeholder_name('path/io'):tx, 
                  tlog.target_output_placeholder_name('path/io'):ty}
      session.run(tlog.inference(mode), feed_dict=train_fd)
      break
    elapsed = time.time() - t0
    print 'learning takes',time.time()-t0,'sec'
    print tx.shape[0],'examples','time',elapsed,'qps',tx.shape[0]/elapsed
    return elapsed

def trainAndTest(tlog,trainDataFile,testDataFile,epochs):
  mode = 'path/io'
  trainData = tlog.load_dataset(trainDataFile)
  testData = tlog.load_dataset(testDataFile)

  predicted_y = tlog.inference(mode)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  unregularized_loss = tlog.loss(mode)
  optimizer = tf.train.AdagradOptimizer(1.0)
  train_step = optimizer.minimize(unregularized_loss)

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  t0 = time.time()
  for i in range(epochs):
    print 'epoch',i+1,'elapsed',time.time()-t0
    for (mode,(tx,ty)) in tlog.minibatches(trainData):
      train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
      session.run(train_step,feed_dict=train_fd)
  print 'learning takes',time.time()-t0,'sec'
  tot_test = 0
  tot_acc = 0
  i = 0
  for (mode,(ux,uy)) in tlog.minibatches(testData):
    i += 1
    m = ux.shape[0] #examples
    test_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
    acc = session.run(accuracy, feed_dict=test_fd)    
    print 'minibatch acc for batch',i,acc
    tot_test += m
    tot_acc += acc*m
  acc = tot_acc/tot_test
  print 'weighted acc',acc
  return acc

def runMain():
  (goal,n,maxD,epochsOrMinibatch) = bigexpt.getargs()
  (factFile,trainFile,testFile) = bigexpt.genInputs(n)
  (tlog,trainData,testData) = setup_tlog(maxD,factFile,trainFile,testFile)
  print 'tlog.prog.maxDepth',tlog.prog.maxDepth
  if goal=='time':
    print timingExpt(tlog,maxD,trainFile,epochsOrMinibatch)
  elif goal=='acc':
    print trainAndTest(tlog,trainFile,testFile,epochsOrMinibatch)
  else:
    assert False,'bad goal %s' % goal

if __name__=="__main__":
  runMain()
