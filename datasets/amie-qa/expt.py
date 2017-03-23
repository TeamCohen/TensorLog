import sys
import getopt
import logging

# tensorflow cross-compilation based experiment

from tensorlog import simple
import tensorflow as tf

class OptHolder(object):
  pass

if __name__=="__main__":
  # parse commandline
  mainArgspec = ["kb_version=", "num_train=", "num_test=", "epochs=", "prog_file="]
  mainOptlist,mainArgs = getopt.getopt(sys.argv[1:], 'x', mainArgspec)
  mainOptdict = dict(mainOptlist)

  c = OptHolder()
  c.kb_version = mainOptdict.get('--kb_version','typed-small')
  c.epochs = int(mainOptdict.get('--epochs','10'))
  c.num_train = int(mainOptdict.get('--num_train','100'))
  c.num_test = int(mainOptdict.get('--num_test','200'))
  c.prog_file = mainOptdict.get('--prog_file','dialog.ppr')
  for (var_name,value) in c.__dict__.items():
    command_line_opt = '--%s' % var_name
    print '# config:',var_name,'=',value,'from',command_line_opt,mainOptdict.get(command_line_opt)

  # create the simple compiler and load the data
  tlog = simple.Compiler(db='idb-%s.cfacts' % c.kb_version,prog=c.prog_file)
  train_data = tlog.load_dataset('train-%d-corpus.exam' % c.num_train)
  test_data = tlog.load_dataset('test-%d-corpus.exam' % c.num_test)

  # check the data is as expected
  mode = 'answer/io'
  assert len(train_data.keys())==1 and mode in train_data
  assert len(test_data.keys())==1 and mode in train_data
  TX,TY = train_data[mode]
  UX,UY = test_data[mode]

  # for evaluating performance
  inference = tlog.inference(mode)
  trueY = tf.placeholder(tf.float32, shape=UY.shape, name='tensorlog/trueY')
  prediction_is_correct = tf.equal(tf.argmax(trueY,1), tf.argmax(inference,1))
  accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))
  test_batch_fd = {tlog.input_placeholder_name(mode):UX, trueY.name:UY}

  # for training
  loss = tlog.loss(mode)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  train_step = optimizer.minimize(loss)
  train_batch_fd = {tlog.input_placeholder_name(mode):TX, tlog.target_output_placeholder_name(mode):TY}

  # do the experiment
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  print 'initial accuracy',session.run(accuracy, feed_dict=test_batch_fd)
  for i in range(c.epochs):
    session.run(train_step, feed_dict=train_batch_fd)
  print 'final accuracy',session.run(accuracy, feed_dict=test_batch_fd)
