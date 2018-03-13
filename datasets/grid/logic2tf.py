import sys
import time
import tensorflow as tf
import numpy as np
import random

FRAC_CROSS_QUADRANT_CONNECTIONS_DROPPED = 1.0
#DEGREE_GIVEN = False
DEGREE_GIVEN = True


#
# simple demo of adding some numeric learning on top of the logic part
# 
# this uses a fixed loss on predictions y of y*(degree - 3.0)^2
# which is like computing degree and then using squared loss against MSE of 3.0
# could do better by having some other regression problem that learns the offset 3.0
#
# better plan:
#  there are features x1,x2 for each node that reveal degree,
#  ie ax1 + bx2 = degree + error
#  to get this, fix a,b let compute x1 ~ N(0,1), error ~ N(0,epsilon)
#  compute x2 = (degree + error - ax1) / b
#  then penalize by (ax1 + bx2 - degree)^2
# 
# x1,x2 are generated but not tested
#
# accuracy is not that great, maybe because the train/test data is out
# of sync with the loss function, not all the corners are correct for
# an example, only the closest one.  maybe I should add a distance
# loss as well?

TARGET_A = 2.0
TARGET_B = 3.0

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
  degree = tlog.db.matEncoding[('degree',1)].todense()
  corner_penalty = np.square(degree - 3.0)
  mode = 'path/io'
  predicted_y = tlog.inference(mode)
  actual_y = tlog.target_output_placeholder(mode)
  correct_predictions = tf.equal(tf.argmax(actual_y,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

  #unregularized_loss = tlog.loss(mode)
  if DEGREE_GIVEN:
    unregularized_loss = tf.reduce_sum(predicted_y * corner_penalty * 10)
  else:
    x1 = tlog.db.matEncoding[('x1',1)].todense()
    x2 = tlog.db.matEncoding[('x2',1)].todense()
    A = tf.Variable(random.random(), name="A")
    B = tf.Variable(random.random(), name="B")
    predicted_degree = A*x1 + B*x2
    predicted_penalty = np.square(predicted_degree - 3.0)
    unregularized_loss = tf.reduce_mean(predicted_y * predicted_penalty)

  optimizer = tf.train.AdagradOptimizer(1.0)
  #optimizer = tf.train.GradientDescentOptimizer(1.0)
  train_step = optimizer.minimize(unregularized_loss)

  session = tf.Session()
  session.run(tf.global_variables_initializer())
  (ux,uy) = testData[mode]
  test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
  acc = session.run(accuracy, feed_dict=test_fd)

  (tx,ty) = trainData[mode]
  train_fd = {tlog.input_placeholder_name(mode):tx, tlog.target_output_placeholder_name(mode):ty}
  print 'initial test acc',acc
  t0 = time.time()
  print 'epoch',
  for i in range(epochs):
    print i+1,
    session.run(train_step, feed_dict=train_fd)
    if (i+1)%3==0:
      test_fd = {tlog.input_placeholder_name(mode):ux, tlog.target_output_placeholder_name(mode):uy}
      train_loss = session.run(unregularized_loss, feed_dict=train_fd)
      train_acc = session.run(accuracy, feed_dict=train_fd)
      test_loss = session.run(unregularized_loss, feed_dict=test_fd)
      test_acc = session.run(accuracy, feed_dict=test_fd)
      if not DEGREE_GIVEN:
        hatA,hatB =  session.run([A,B], feed_dict=train_fd)
        print 'model A,B',hatA,hatB,
      print 'train loss',train_loss,'acc',train_acc,'test loss',test_loss,'acc',test_acc
      print 'epoch',
  print 'done'
  print 'learning takes',time.time()-t0,'sec'

  acc = session.run(accuracy, feed_dict=test_fd)
  print 'test acc',acc

  tlog.set_all_db_params_to_learned_values(session)
  tlog.serialize_db('learned.db')
  return acc

def augmentFacts(inFactFile,n):
  outFactfile = inFactFile[:-len(".cfacts")] + "_plus_deg.cfacts"
  nskip = 0
  nline = 0
  with open(outFactfile,'w') as outFp:
    #copy old facts
    for line in open(inFactFile):
      nline += 1
      # edge i1,j1 i2,j2 weight
      parts = line.strip().split("\t")
      i1,j1 = map(int,parts[1].split(","))
      i2,j2 = map(int,parts[2].split(","))
      if (i1==n/2 and i2==(n/2+1)) or (j1==n/2 and j2==(n/2+1)) and random.random()<=FRAC_CROSS_QUADRANT_CONNECTIONS_DROPPED:
        #print 'skip',i1,j1,'->',i2,j2,line,
        nskip += 1
      else:
        outFp.write(line)
    print 'skipped',nskip,'of',nline,'edges'

    # compute and write degree
    def extreme(k): return k==1 or k==n
    degree = {}
    for i in range(1,n+1):
      for j in range(1,n+1):
        v = "%d,%d" % (i,j)
        if extreme(i) and extreme(j): 
          degree[v] = 3.0
        elif extreme(i) or extreme(j): 
          degree[v] = 5.0
        else: 
          degree[v] = 8.0
        outFp.write("\t".join(["degree",v,"%.1f" % degree[v]]) + '\n')
  
    # compute features x1,x2 which predict degree, approximately, via A*x1 + B*X2 = degree + noise
    x2 = {}
    r = random.Random()
    for v in degree:
      noisy_d = degree[v] + r.gauss(0.0, 1.0)
      x1 = r.gauss(1.0, 1.0)
      x2[v] = (noisy_d - TARGET_A*x1) / TARGET_B
      outFp.write("\t".join(["x1",v,"%.2f" % x1]) + '\n')
    for v in x2:
      outFp.write("\t".join(["x2",v,"%.2f" % x2[v]]) + '\n')
  return outFactfile

def runMain():
  (goal,n,maxD,epochs) = expt.getargs()
  assert goal=="acc"
  (factFile,trainFile,testFile) = expt.genInputs(n)
  factFile = augmentFacts(factFile,n)
  print 'using factFile',factFile,'n',n
  (tlog,trainData,testData) = setup_tlog(maxD,factFile,trainFile,testFile)
  trainAndTest(tlog,trainData,testData,epochs)

if __name__=="__main__":
  runMain()
